from typing import Dict, Tuple, Any, List
import numpy as np
import cv2
import os
import sys
import time
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from functools import partial
from threading import Thread

# 导入 OpenPose 库
try:
    # 尝试从系统路径导入
    sys.path.append('/usr/local/python')
    from openpose import pyopenpose as op
except ImportError as e:
    print('无法导入 OpenPose 库，请设置系统路径。')
    raise e


class ActionRecognitionGUI(QWidget):
    def __init__(self):
        super().__init__()

        # 设置窗口标题和大小
        self.setWindowTitle('动作识别')
        self.resize(800, 600)

        # 创建控件
        self.video_label = QLabel(self)
        self.start_button = QPushButton('开始', self)
        self.progress_bar = QProgressBar(self)
        self.status_label = QLabel('等待中', self)

        # 创建布局
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addWidget(self.start_button)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.status_label)
        self.setLayout(main_layout)

        # 连接信号和槽
        self.start_button.clicked.connect(self.start_recognition)

        # 初始化 OpenPose
        self.op_wrapper = op.WrapperPython()
        params = dict(model_folder='openpose/models')
        self.op_wrapper.configure(params)
        self.op_wrapper.start()

        # 初始化识别器
        self.recognizer = ActionRecognizer()

    def start_recognition(self):
        # 获取视频路径
        file_path, _ = QFileDialog.getOpenFileName(self, '选择视频文件', '.', '视频文件 (*.mp4 *.avi *.mov)')
        if not file_path:
            return

        # 启动识别线程
        self.recognition_thread = ActionRecognitionThread(self.recognizer, file_path)
        self.recognition_thread.result_signal.connect(self.update_progress)
        self.recognition_thread.start()

        # 显示视频
        self.video_cap = cv2.VideoCapture(file_path)
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.fps = int(self.video_cap.get(cv2.CAP_PROP_FPS))
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.show_frame)
        self.timer.start(1000 // self.fps)

        # 更新界面状态
        self.status_label.setText('正在识别视频...')
        self.start_button.setEnabled(False)
        self.progress_bar.setValue(0)

    def show_frame(self):
        ret, frame = self.video_cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, c = frame.shape
            qimage = QImage(frame, w, h, w * c, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qimage))
        else:
            self.timer.stop()
            self.video_cap.release()
            self.video_label.clear()
            self.status_label.setText('等待中')
            self.start_button.setEnabled(True)

    def update_progress(self, progress):
        self.progress_bar.setValue(progress)
        if progress == 100:
            self.status_label.setText('识别完成')
            self.start_button.setEnabled(True)


class ActionRecognitionThread(QThread):
    result_signal = pyqtSignal(int)

    def __
