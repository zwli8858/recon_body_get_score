from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.uic import loadUi
import cv2
import numpy as np
import time
from openpose import pyopenpose as op


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi('mainwindow.ui', self)
        self.startButton.clicked.connect(self.start_recognition)
        self.openButton.clicked.connect(self.open_file_dialog)
        self.video_player = VideoPlayer(self.videoLabel)

        self.params = dict()
        self.params["model_folder"] = "./openpose/models/"
        self.params["model_pose"] = "BODY_25"

    def open_file_dialog(self):
        # 打开文件对话框，选择要播放的视频
        file_dialog = QFileDialog()
        file_path = file_dialog.getOpenFileName(self, '选择文件', './')
        if file_path[0]:
            self.video_player.load_video(file_path[0])
            self.startButton.setEnabled(True)

    def start_recognition(self):
        # 禁用开始按钮，避免多次点击
        self.startButton.setEnabled(False)

        # 开始进行动作识别
        score = 0
        start_time = time.time()
        frame_count = 0
        while True:
            # 读取视频帧
            ret, frame = self.video_player.get_frame()
            if not ret:
                break

            # 进行动作识别
            pose_keypoints, frame = self.recognition(frame)

            # 计算动作分数
            if pose_keypoints is not None:
                score += compute_score(pose_keypoints)
                frame_count += 1

            # 将帧显示在界面上
            self.video_player.display_frame(frame)

        # 计算平均分数
        if frame_count > 0:
            score /= frame_count

        # 显示动作分数
        self.scoreLabel.setText('%.2f' % score)

        # 重新启用开始按钮
        self.startButton.setEnabled(True)

        # 打印动作识别耗时
        print('Recognition time:', time.time() - start_time)

    def recognition(self, frame):
        # 初始化OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(self.params)
        opWrapper.start()

        # 运行OpenPose
        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])

        # 获取姿态关键点
        pose_keypoints = None
        if len(datum.poseKeypoints) > 0:
            pose_keypoints = datum.poseKeypoints[0]

        # 可视化结果
        if len(datum.poseKeypoints) > 0:
            frame = datum.cvOutputData

        return pose_keypoints, frame


class VideoPlayer:
    def __init__(self, label):
        self.label = label
        self.cap = None

    def load_video(self, file_path):
        self.cap = cv2.VideoCapture(file_path)

    def get_frame(self):
        if self.cap is None:
            return None, None
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        return
