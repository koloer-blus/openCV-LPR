"""
基于 OpenCV 的车牌识别（新方案）

相对原 predict.py 的主要变化：
1. 颜色优先定位（蓝/黄/绿 HSV 掩膜）+ 边缘形态学兜底
2. 透视矫正（minAreaRect + warpPerspective）
3. 连通域 + 垂直投影混合切分字符
4. 复用 EasyPR 风格 HOG+SVM 模型识别中文/字母数字
5. 多候选评分，避免大色块（车身/卡车厢）抢走车牌

用法：
  python lpr_opencv.py test/car3.jpg
  python lpr_opencv.py --test-dir test
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from numpy.linalg import norm

SZ = 20
MAX_SIDE = 1200
MIN_PLATE_AREA = 600

PROVINCES = [
    "zh_cuan", "川",
    "zh_e", "鄂",
    "zh_gan", "赣",
    "zh_gan1", "甘",
    "zh_gui", "贵",
    "zh_gui1", "桂",
    "zh_hei", "黑",
    "zh_hu", "沪",
    "zh_ji", "冀",
    "zh_jin", "津",
    "zh_jing", "京",
    "zh_jl", "吉",
    "zh_liao", "辽",
    "zh_lu", "鲁",
    "zh_meng", "蒙",
    "zh_min", "闽",
    "zh_ning", "宁",
    "zh_qing", "靑",
    "zh_qiong", "琼",
    "zh_shan", "陕",
    "zh_su", "苏",
    "zh_sx", "晋",
    "zh_wan", "皖",
    "zh_xiang", "湘",
    "zh_xin", "新",
    "zh_yu", "豫",
    "zh_yu1", "渝",
    "zh_yue", "粤",
    "zh_yun", "云",
    "zh_zang", "藏",
    "zh_zhe", "浙",
]
PROVINCE_START = 1000

# HSV 范围（OpenCV H:0-180）。阈值偏松，靠后续评分过滤。
COLOR_RANGES = {
    "blue": [
        (np.array([90, 30, 40]), np.array([140, 255, 255])),
    ],
    "yellow": [
        (np.array([11, 50, 50]), np.array([40, 255, 255])),
    ],
    # 新能源绿牌常为低饱和浅绿
    "green": [
        (np.array([35, 15, 50]), np.array([95, 255, 255])),
    ],
}


def imread_unicode(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return img


def order_box_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_plate(img: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
    ordered = order_box_points(box)
    tl, tr, br, bl = ordered
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if width < 40 or height < 12:
        return None
    # 统一车牌朝向：宽 > 高
    if height > width * 1.2:
        return None
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(img, matrix, (width, height))


def deskew(img: np.ndarray) -> np.ndarray:
    m = cv2.moments(img)
    if abs(m["mu02"]) < 1e-2:
        return img.copy()
    skew = m["mu11"] / m["mu02"]
    matrix = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    return cv2.warpAffine(
        img, matrix, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    )


def preprocess_hog(digits: List[np.ndarray]) -> np.ndarray:
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bins = np.int32(bin_n * ang / (2 * np.pi))
        bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [
            np.bincount(b.ravel(), m.ravel(), bin_n)
            for b, m in zip(bin_cells, mag_cells)
        ]
        hist = np.hstack(hists)
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps
        samples.append(hist)
    return np.float32(samples)


class SVMModel:
    def __init__(self, c: float = 1.0, gamma: float = 0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(c)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def load(self, path: str) -> None:
        self.model = self.model.load(path)

    def predict(self, samples: np.ndarray) -> np.ndarray:
        return self.model.predict(samples)[1].ravel()


@dataclass
class PlateCandidate:
    score: float
    color: str
    plate: np.ndarray
    box: np.ndarray
    peaks: int
    purity: float


class LicensePlateRecognizer:
    def __init__(self, model_dir: str = "."):
        self.model_dir = model_dir
        self.model = SVMModel()
        self.model_chinese = SVMModel()
        self._load_models()

    def _load_models(self) -> None:
        svm_path = os.path.join(self.model_dir, "svm.dat")
        svm_cn_path = os.path.join(self.model_dir, "svmchinese.dat")
        if not os.path.exists(svm_path) or not os.path.exists(svm_cn_path):
            raise FileNotFoundError(
                "缺少 svm.dat / svmchinese.dat，请先放到项目根目录"
            )
        self.model.load(svm_path)
        self.model_chinese.load(svm_cn_path)

    def predict(self, image_or_path) -> Tuple[List[str], Optional[np.ndarray], Optional[str]]:
        img = (
            imread_unicode(image_or_path)
            if isinstance(image_or_path, (str, Path))
            else image_or_path
        )
        candidates = self._locate_plates(img)
        for cand in candidates:
            chars = self._recognize_plate(cand.plate, cand.color)
            if 6 <= len(chars) <= 8:
                return chars, cand.plate, cand.color
        return [], None, None

    # ---------------- localization ----------------
    def _locate_plates(self, img: np.ndarray) -> List[PlateCandidate]:
        h, w = img.shape[:2]
        scale = MAX_SIDE / max(h, w) if max(h, w) > MAX_SIDE else 1.0
        small = (
            cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            if scale != 1.0
            else img
        )

        boxes: List[Tuple[str, np.ndarray, float]] = []
        boxes.extend(self._color_boxes(small, scale))
        boxes.extend(self._edge_boxes(small, scale))

        candidates: List[PlateCandidate] = []
        seen = []
        for color, box, area in boxes:
            if any(self._iou(box, other) > 0.55 for other in seen):
                continue
            plate = warp_plate(img, box)
            if plate is None:
                continue
            # 统一缩放到便于切分的高度
            ph, pw = plate.shape[:2]
            target_h = 64
            plate = cv2.resize(
                plate,
                (max(120, int(pw * target_h / ph)), target_h),
                interpolation=cv2.INTER_CUBIC,
            )
            metrics = self._plate_metrics(plate, color)
            if metrics is None:
                continue
            peaks, fill, purity, ratio = metrics
            y_center = float(np.mean(box[:, 1])) / h
            score = (
                purity * 2.2
                + (1.0 - abs(peaks - 7.5) / 7.5) * 1.4
                + (1.0 - abs(ratio - 3.5) / 3.5) * 0.6
                + (1.0 - abs(y_center - 0.68)) * 0.35
                + min(area / 8000.0, 1.0) * 0.2
            )
            # 填充率合理加分
            score += max(0.0, 1.0 - abs(fill - 0.32) / 0.32) * 0.5
            candidates.append(
                PlateCandidate(score, color, plate, box, peaks, purity)
            )
            seen.append(box)

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:8]

    def _color_boxes(
        self, small: np.ndarray, scale: float
    ) -> List[Tuple[str, np.ndarray, float]]:
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        results = []
        for color, ranges in COLOR_RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, lo, hi)
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 5))
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < MIN_PLATE_AREA:
                    continue
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                rw, rh = rect[1]
                if rw < rh:
                    rw, rh = rh, rw
                if rh < 8:
                    continue
                ratio = rw / max(rh, 1e-6)
                if not (2.0 <= ratio <= 6.0):
                    continue
                results.append((color, box / scale, float(area / (scale * scale))))
        return results

    def _edge_boxes(
        self, small: np.ndarray, scale: float
    ) -> List[Tuple[str, np.ndarray, float]]:
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        kernel = np.ones((18, 18), np.uint8)
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        enhanced = cv2.addWeighted(gray, 1, opening, -1, 0)
        _, thr = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edge = cv2.Canny(thr, 100, 200)
        morph = cv2.morphologyEx(
            edge, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        )
        morph = cv2.morphologyEx(
            morph, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        )
        contours, _ = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_PLATE_AREA:
                continue
            rect = cv2.minAreaRect(cnt)
            rw, rh = rect[1]
            if rw < rh:
                rw, rh = rh, rw
            if rh < 8:
                continue
            ratio = rw / max(rh, 1e-6)
            if not (2.2 <= ratio <= 5.8):
                continue
            box = cv2.boxPoints(rect) / scale
            # 用裁剪区域主色估计颜色
            plate = warp_plate(
                cv2.resize(
                    small,
                    (int(small.shape[1] / scale), int(small.shape[0] / scale)),
                )
                if scale != 1
                else small,
                box * scale if scale != 1 else box,
            )
            # 上面 scale 处理太绕，直接用原图尺寸 box 在后续 warp；这里仅标 color=unknown 再复判
            color = self._guess_color_from_box(small, box * scale)
            results.append((color, box, float(area / (scale * scale))))
        return results

    def _guess_color_from_box(self, small: np.ndarray, box_on_small: np.ndarray) -> str:
        plate = warp_plate(small, box_on_small)
        if plate is None:
            return "blue"
        hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        blue = ((h >= 90) & (h <= 140) & (s > 30)).mean()
        yellow = ((h >= 11) & (h <= 40) & (s > 50)).mean()
        green = ((h >= 35) & (h <= 95) & (s > 15) & (v > 50)).mean()
        scores = {"blue": blue, "yellow": yellow, "green": green}
        return max(scores, key=scores.get)

    def _plate_metrics(
        self, plate: np.ndarray, color: str
    ) -> Optional[Tuple[int, float, float, float]]:
        h, w = plate.shape[:2]
        ratio = w / max(h, 1)
        if not (2.0 <= ratio <= 6.2):
            return None

        hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
        hh, ss, vv = cv2.split(hsv)
        if color == "blue":
            purity = ((hh >= 90) & (hh <= 140) & (ss > 25)).mean()
        elif color == "yellow":
            purity = ((hh >= 11) & (hh <= 40) & (ss > 40)).mean()
        else:
            purity = ((hh >= 35) & (hh <= 95) & (ss > 10) & (vv > 40)).mean()
        if purity < 0.12:
            return None

        binary = self._plate_binary(plate, color)
        # 去掉上下白边
        row_sum = (binary > 0).sum(axis=1)
        if row_sum.max() == 0:
            return None
        active = np.where(row_sum > row_sum.max() * 0.15)[0]
        if len(active) < 8:
            return None
        binary = binary[active[0] : active[-1] + 1]
        proj = (binary > 0).sum(axis=0).astype(np.float32)
        peaks = self._count_peaks(proj, thr_ratio=0.22)
        fill = float((binary > 0).mean())
        if not (4 <= peaks <= 12):
            return None
        if not (0.12 <= fill <= 0.70):
            return None
        return peaks, fill, float(purity), float(ratio)

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        ax1, ay1 = a.min(axis=0)
        ax2, ay2 = a.max(axis=0)
        bx1, by1 = b.min(axis=0)
        bx2, by2 = b.max(axis=0)
        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        return inter / max(area_a + area_b - inter, 1e-6)

    @staticmethod
    def _count_peaks(proj: np.ndarray, thr_ratio: float = 0.25) -> int:
        if proj.size == 0 or proj.max() < 1:
            return 0
        thr = proj.max() * thr_ratio
        peaks = 0
        in_peak = False
        for v in proj:
            if not in_peak and v >= thr:
                in_peak = True
                peaks += 1
            elif in_peak and v < thr:
                in_peak = False
        return peaks

    @staticmethod
    def _plate_binary(plate: np.ndarray, color: str) -> np.ndarray:
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        # 蓝牌：白字；黄/绿牌：黑字，需反色后统一成白字黑底
        if color in ("yellow", "green"):
            gray = cv2.bitwise_not(gray)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    # ---------------- recognition ----------------
    def _recognize_plate(self, plate: np.ndarray, color: str) -> List[str]:
        binary = self._plate_binary(plate, color)
        h, w = binary.shape
        # 水平裁剪字符行
        row_sum = (binary > 0).sum(axis=1)
        thr = row_sum.max() * 0.2
        ys = np.where(row_sum >= thr)[0]
        if len(ys) == 0:
            return []
        y0, y1 = ys[0], ys[-1]
        binary = binary[y0 : y1 + 1, :]
        # 去掉左右边缘噪声
        col_sum = (binary > 0).sum(axis=0)
        xs = np.where(col_sum >= col_sum.max() * 0.08)[0]
        if len(xs) == 0:
            return []
        binary = binary[:, xs[0] : xs[-1] + 1]

        parts = self._segment_chars(binary)
        if len(parts) < 6:
            return []

        # 新能源绿牌通常 8 位；普通 7 位
        expect = 8 if color == "green" else 7
        if len(parts) > expect + 1:
            # 去掉过窄噪声
            widths = [p.shape[1] for p in parts]
            med = np.median(widths)
            parts = [p for p in parts if p.shape[1] >= med * 0.35]
        if len(parts) > expect:
            # 优先丢掉最窄的多余块（分隔点/铆钉）
            while len(parts) > expect:
                idx = int(np.argmin([p.shape[1] for p in parts]))
                # 尽量不删第一个中文位
                if idx == 0 and len(parts) > 1:
                    idx = int(np.argmin([p.shape[1] for p in parts[1:]])) + 1
                parts.pop(idx)

        result = []
        for i, part in enumerate(parts):
            if part.mean() < 255 / 6:
                continue
            char_img = self._normalize_char(part)
            feat = preprocess_hog([deskew(char_img)])
            if i == 0:
                resp = self.model_chinese.predict(feat)
                idx = int(resp[0]) - PROVINCE_START
                if idx < 0 or idx >= len(PROVINCES):
                    continue
                ch = PROVINCES[idx]
            else:
                resp = self.model.predict(feat)
                ch = chr(int(resp[0]))
            # 末尾细长噪声常被识别成 1
            if ch == "1" and i == len(parts) - 1 and part.shape[0] / max(part.shape[1], 1) >= 7:
                continue
            result.append(ch)
        return result

    def _segment_chars(self, binary: np.ndarray) -> List[np.ndarray]:
        """连通域优先，失败则退回垂直投影。"""
        h, w = binary.shape
        # 轻微腐蚀，拆开粘连
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        work = cv2.erode(binary, kernel, iterations=1)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
        boxes = []
        for i in range(1, num):
            x, y, bw, bh, area = stats[i]
            if area < h * 0.8:
                continue
            if bh < h * 0.35 or bw < 2:
                continue
            if bw > w * 0.45:
                continue
            aspect = bh / max(bw, 1)
            if aspect < 0.7:  # 太扁，多半是铆钉/分隔点
                continue
            boxes.append((x, y, bw, bh))
        boxes.sort(key=lambda b: b[0])

        parts = []
        if 6 <= len(boxes) <= 10:
            # 合并被拆开的中文省份（第一个字符可能被切成多块）
            if len(boxes) >= 8:
                first = boxes[0]
                second = boxes[1]
                if first[2] + second[2] < h * 0.9 and second[0] - (first[0] + first[2]) < h * 0.25:
                    x = first[0]
                    y = min(first[1], second[1])
                    r = max(first[0] + first[2], second[0] + second[2])
                    b = max(first[1] + first[3], second[1] + second[3])
                    boxes = [(x, y, r - x, b - y)] + boxes[2:]
            for x, y, bw, bh in boxes:
                pad = 1
                x0 = max(0, x - pad)
                x1 = min(w, x + bw + pad)
                parts.append(binary[:, x0:x1])
            return parts

        # 投影切分兜底
        proj = (binary > 0).sum(axis=0).astype(np.float32)
        thr = max(proj.max() * 0.18, 1.0)
        waves = []
        start = None
        for i, v in enumerate(proj):
            if start is None and v >= thr:
                start = i
            elif start is not None and v < thr:
                if i - start >= 2:
                    waves.append((start, i))
                start = None
        if start is not None and w - start >= 2:
            waves.append((start, w))

        if not waves:
            return []

        # 合并过窄的中文碎块
        max_w = max(b - a for a, b in waves)
        merged = []
        i = 0
        while i < len(waves):
            a, b = waves[i]
            while i + 1 < len(waves) and (b - a) < max_w * 0.55:
                i += 1
                b = waves[i][1]
                if (b - a) >= max_w * 0.55:
                    break
            merged.append((a, b))
            i += 1

        # 去掉分隔圆点
        if len(merged) >= 3:
            a, b = merged[2]
            if (b - a) < max_w / 3 and binary[:, a:b].mean() < 255 / 5:
                merged.pop(2)

        return [binary[:, a:b] for a, b in merged]

    @staticmethod
    def _normalize_char(part: np.ndarray) -> np.ndarray:
        h, w = part.shape
        # 去掉上下空白
        rows = np.where(part.sum(axis=1) > 0)[0]
        cols = np.where(part.sum(axis=0) > 0)[0]
        if len(rows) and len(cols):
            part = part[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
        pad = max(1, part.shape[1] // 3)
        part = cv2.copyMakeBorder(
            part, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0
        )
        return cv2.resize(part, (SZ, SZ), interpolation=cv2.INTER_AREA)


# 测试集人工标注（用于 --test-dir 准确率）
TEST_GROUND_TRUTH = {
    "1.jpg": "京E51619",
    "2.jpg": "京AD77972",
    "car3.jpg": "鲁Q521MZ",
    "car4.jpg": "吉AA266G",
    "car5.jpg": "京AG6104",
    "car7.jpg": "豫C66666",
    "lLD9016.jpg": "鲁LD9016",
    "wA87271.jpg": "皖A87271",
    "wATH859.jpg": "皖ATH859",
    "wAUB816.jpg": "皖AUB816",
}


def run_on_image(recognizer: LicensePlateRecognizer, path: str, save_roi: Optional[str] = None):
    chars, roi, color = recognizer.predict(path)
    text = "".join(chars) if chars else ""
    print(f"{path}: color={color} result={text or None}")
    if save_roi and roi is not None:
        out = Path(save_roi)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".jpg", roi)[1].tofile(str(out))
    return text, color


def run_test_dir(recognizer: LicensePlateRecognizer, test_dir: str):
    paths = sorted(Path(test_dir).glob("*.jpg"))
    ok = 0
    for p in paths:
        text, color = run_on_image(
            recognizer, str(p), save_roi=str(Path("debug_plates_v2") / p.name)
        )
        gt = TEST_GROUND_TRUTH.get(p.name)
        if gt is not None:
            match = text == gt
            ok += int(match)
            print(f"  GT={gt}  {'OK' if match else 'MISS'}")
    print(f"\n准确率: {ok}/{len(paths)}")


def main():
    parser = argparse.ArgumentParser(description="OpenCV 车牌识别新方案")
    parser.add_argument("image", nargs="?", help="单张图片路径")
    parser.add_argument("--test-dir", default=None, help="批量测试目录")
    parser.add_argument("--model-dir", default=".", help="svm.dat 所在目录")
    args = parser.parse_args()

    recognizer = LicensePlateRecognizer(model_dir=args.model_dir)
    if args.test_dir:
        run_test_dir(recognizer, args.test_dir)
    elif args.image:
        run_on_image(recognizer, args.image)
    else:
        run_test_dir(recognizer, "test")


if __name__ == "__main__":
    main()
