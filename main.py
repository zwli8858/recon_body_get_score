import cv2
import numpy as np
import time
import os
import argparse

# 导入OpenPose库
from openpose import pyopenpose as op

# 定义动作分类模型和标签
from sklearn.svm import SVC

model = SVC(kernel='rbf', gamma='auto', C=1)
labels = {0: 'walking', 1: 'running', 2: 'jumping'}

# 定义OpenPose参数
params = dict()
params["model_folder"] = "models/"
params["model_pose"] = "COCO"
params["net_resolution"] = "-1x368"
params["output_resolution"] = "-1x-1"
params["num_gpu_start"] = 0
params["disable_blending"] = False
params["default_model_folder"] = "models/"
params["render_threshold"] = 0.05
params["alpha_pose"] = 0.6
params["scale_gap"] = 0.3
params["scale_number"] = 1
params["disable_multi_thread"] = False
params["fps_max"] = 30.0

# 初始化OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# 读取视频文件
cap = cv2.VideoCapture('video.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 定义变量记录动作信息
action_start_time = None
last_label = None
speeds = []

# 处理每一帧
while True:
    # 读取一帧
    ret, frame = cap.read()
    if not ret:
        break

    # 转换颜色空间
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 运行OpenPose
    datum = op.Datum()
    datum.cvInputData = img
    opWrapper.emplaceAndPop([datum])
    keypoints = datum.poseKeypoints

    # 如果检测到人体关键点，进行动作分类和时间计算
    if keypoints is not None:
        # 提取关键点坐标
        x = keypoints[0][:, 0]
        y = keypoints[0][:, 1]

        # 对关键点序列进行归一化处理
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))

        # 进行动作分类
        feature = np.hstack((x, y))
        label = model.predict([feature])[0]

        # 如果检测到新的动作，记录开始时间戳
        if label != last_label:
            action_start_time = time.time()
            last_label = label

            # 如果当前动作已经持续一定时间，计算速度并记录
            if action_start_time is not None and time.time() - action_start_time > 2.0:
                action_duration = time.time() - action_start_time
                speed = np.sum(np.abs(np.diff(x))) / action_duration
                speeds.append(speed)

                print('Action:', labels[label])
                print('Duration:', action_duration)
                print('Speed:', speed)
                print('---')
                action_start_time = None

    # 计算平均速度和评估动作质量
    avg_speed = np.mean(speeds)
    quality_score = np.interp(avg_speed, [0, 10], [0, 100])
    print('Average speed:', avg_speed)
    print('Quality score:', quality_score)

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

