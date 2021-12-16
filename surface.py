import tkinter as tk
from tkinter.filedialog import *
from tkinter import ttk
import predict
import cv2
from PIL import Image, ImageTk
import threading
import time


class Surface(ttk.Frame):
    pic_path = ""
    view_height = 400
    view_width = 400
    update_time = 0
    thread = None
    thread_run = False
    camera = None
    color_transform = {"green": ("绿牌", "#55FF55"), "yello": ("黄牌", "#FFFF00"), "blue": ("蓝牌", "#6666FF")}

    # 初始化
    def __init__(self, win):
        ttk.Frame.__init__(self, win)
        img_area = ttk.Frame(self)
        analysis_img_info = ttk.Frame(self)
        analysis_res_data = ttk.Frame(self)
        # 设置Header
        header_label = tk.Label(
            win,
            text='基于OpenCV的SVM算法实现的车牌识别',
            bg='#0088FF',
            font=('Arial', 16),
            fg="#fff", height=2
        )
        header_label.pack()

        win.title("OpenCV LPR")
        win.state("zoomed")

        self.pack(fill=tk.BOTH, expand=tk.YES, padx="5", pady="5")
        img_area.pack(side=tk.LEFT, expand=1, fill=tk.BOTH)
        analysis_img_info.pack(side=tk.TOP, expand=1, fill=tk.Y)
        analysis_res_data.pack(side=tk.RIGHT, expand=0)

        tk.Label(
            img_area,
            text='需要进行车牌提取的图片：',
            font=('Arial', 16),
        ).pack(anchor="nw")

        tk.Label(
            analysis_img_info,
            text='车牌截取图片：',
            font=('Arial', 16),
        ).grid(column=0, row=0, sticky=tk.W)

        from_pic_ctl = tk.Button(
            analysis_res_data,
            text="选择图片",
            width=20,
            height=2,
            activebackground="#fff",
            activeforeground="#0088FF",
            font=('Arial', 16),
            bg="#fff",
            command=self.from_pic
        )
        self.image_ctl = ttk.Label(img_area)
        self.image_ctl.pack(anchor="nw")

        self.roi_ctl = ttk.Label(analysis_img_info)
        self.roi_ctl.grid(column=0, row=1, sticky=tk.W)

        ttk.Label(
            analysis_img_info,
            text='识别结果：',
            font=('Arial', 16),
        ).grid(column=0, row=8, sticky=tk.W)

        self.r_ctl = ttk.Label(
            analysis_img_info,
            font=('Arial', 16),
            text=""
        )
        self.r_ctl.grid(column=0, row=12, sticky=tk.W)

        self.color_ctl = ttk.Label(
            analysis_img_info,
            text="",
            font=('Arial', 16),
            width="20",
        )
        self.color_ctl.grid(column=0, row=16, sticky=tk.W)
        from_pic_ctl.pack(anchor="se", pady="5")
        self.predictor = predict.CardPredictor()
        self.predictor.train_svm()
    # 图片展示
    def get_imgtk(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im)
        wide = imgtk.width()
        high = imgtk.height()
        if wide > self.view_width or high > self.view_height:
            wide_factor = self.view_width / wide
            high_factor = self.view_height / high
            factor = min(wide_factor, high_factor)

            wide = int(wide * factor)
            if wide <= 0: wide = 1
            high = int(high * factor)
            if high <= 0: high = 1
            im = im.resize((wide, high), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(image=im)
        return imgtk

    # 展示结果
    def show_roi(self, r, roi, color):
        if r:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = Image.fromarray(roi)
            self.imgtk_roi = ImageTk.PhotoImage(image=roi)
            self.roi_ctl.configure(image=self.imgtk_roi, state='enable')

            # 控制车牌号的展示
            self.r_ctl.configure(text="".join(r))
            self.update_time = time.time()

            # 控制结果展示以及背景色控制
            try:
                c = self.color_transform[color]
                self.color_ctl.configure(text=c[0], background=c[1], state='enable')
            except:
                self.color_ctl.configure(state='disabled')
        elif self.update_time + 8 < time.time():
            self.roi_ctl.configure(state='disabled')
            self.r_ctl.configure(text="")
            self.color_ctl.configure(state='disabled')

    # 选择图片
    def from_pic(self):
        self.thread_run = False
        self.pic_path = askopenfilename(title="选择识别图片", filetypes=[("jpg图片", "*.jpg")])
        if self.pic_path:
            img_bgr = predict.imreadex(self.pic_path)
            self.imgtk = self.get_imgtk(img_bgr)
            self.image_ctl.configure(image=self.imgtk)
            resize_rates = (1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4)
            for resize_rate in resize_rates:
                print("resize_rate:", resize_rate)
                r, roi, color = self.predictor.predict(img_bgr, resize_rate)
                if r:
                    break
            # r, roi, color = self.predictor.predict(img_bgr, 1)
            self.show_roi(r, roi, color)

    @staticmethod
    def vedio_thread(self):
        self.thread_run = True
        predict_time = time.time()
        while self.thread_run:
            _, img_bgr = self.camera.read()
            self.imgtk = self.get_imgtk(img_bgr)
            self.image_ctl.configure(image=self.imgtk)
            if time.time() - predict_time > 2:
                r, roi, color = self.predictor.predict(img_bgr)
                self.show_roi(r, roi, color)
                predict_time = time.time()
        print("run end")


def close_window():
    print("destroy")
    if surface.thread_run:
        surface.thread_run = False
        surface.thread.join(2.0)
    win.destroy()


if __name__ == '__main__':
    win = tk.Tk()

    surface = Surface(win)
    win.protocol('WM_DELETE_WINDOW', close_window)
    win.mainloop()
