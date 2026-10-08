#!/usr/bin/env sh

# 发生错误时终止
set -e

pip install numpy
pip install opencv-python
python3 -m pip install --upgrade Pillow
pip install --upgrade Pillow
# 粗定位 + OCR 方案依赖
pip install rapidocr-onnxruntime