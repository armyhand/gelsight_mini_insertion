import cv2
import numpy as np
import time
import socket
from time import sleep

def get_raw_img(frame, img_width, img_height):
    img = cv2.resize(frame, (895, 672))  # size suggested by janos to maintain aspect ratio
    border_size_x, border_size_y = int(img.shape[0] * (1 / 7)), int(
        np.floor(img.shape[1] * (1 / 7)))  # remove 1/7th of border from each size
    img = img[border_size_x:img.shape[0] - border_size_x, border_size_y:img.shape[1] - border_size_y]
    img = img[:, :-1]  # remove last column to get a popular image resolution
    img = cv2.resize(img, (img_width, img_height))  # final resize for 3d

    return img


#捕获摄像头帧
# cap1 = cv2.VideoCapture(0)
# cv2.namedWindow('gelsight_left')
#
# v_fps = 25 # 图片的采集频率为25赫兹,视频保持与其相同
# c_fps = 5 # 每5帧截一张图
# width = 640
# height = 480
#
# videoWriter1 = cv2.VideoWriter('first time/during.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), v_fps,
#                                   (width, height))
# k = 0
# while True:
#     t1 = time.time()
#     success1, frame1 = cap1.read()
#     if success1:
#         frame1 = get_raw_img(frame1, width, height)
#     videoWriter1.write(frame1)
#     cv2.imshow('gelsight_left', frame1)
#
#     if success1 and k % c_fps == 0:
#         status = cv2.imwrite(r"first time/" + str(k) + '.jpg', frame1)
#         # status = cv2.imwrite(r"second time/" + str(k) + '.jpg', frame2)
#
#     k += 1
#     if k % 100 == 0:
#         print("cut " + str(k))
#
#     t2 = time.time()
#     print('freq=', 1.0 / (t2 - t1))
#     if cv2.waitKey(1) == ord("q"):
#         break
#
# cv2.destroyWindow('gelsight_left')
#
# cap1.release()

# 相机参数设置
def Setcamera(cap):
    cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    cap.set(3, 480)
    cap.set(4, 640)

# # socket.SOCK_STREAM 代表基于TCP的流式socket通信
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# # 连接服务端
# ip_port=('localhost', 30001)#进行连接服务器
# sock.connect(ip_port)

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(r'first time/wujiaoxing.avi')
Setcamera(cap)

# 每0.1S计算一次帧率
t = 0.1
counter = 0
fps = 0
start_time = time.time()

v_fps = 25 # 图片的采集频率为25赫兹,视频保持与其相同
c_fps = 5 # 每5帧截一张图
width = 640
height = 480
videoWriter1 = cv2.VideoWriter('first time/during.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), v_fps,
                                  (width, height))

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 25]
ret, frame = cap.read()
while ret:
    frame = get_raw_img(frame, 640, 480)
    videoWriter1.write(frame)

    # 测帧率
    counter += 1
    if (time.time() - start_time) > t:
        fps = counter / (time.time() - start_time)
        fps = str(fps)
        counter = 0
        start_time = time.time()
    cv2.putText(frame, "FPS {0}".format(fps), (10, 30), 1, 1.5, (255, 0, 255), 2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
# import numpy as np
# from scipy.optimize import minimize
#
#
# def compute_overlap(array1, array2):
#     # 计算每组坐标数组的包络
#     envelope1 = np.column_stack((array1.min(axis=0), array1.max(axis=0)))
#     envelope2 = np.column_stack((array2.min(axis=0), array2.max(axis=0)))
#
#     # 定义重叠面积的目标函数
#
#     def overlap_area(shift):
#         shifted_envelope2 = envelope2 + shift
#         intersection = np.column_stack(
#             (np.maximum(envelope1[:, 0], shifted_envelope2[:, 0]), np.minimum(envelope1[:, 1], shifted_envelope2[:, 1])))
#         area_intersection = np.prod(np.maximum(0, intersection[:, 1] - intersection[:, 0]))
#         area_envelope1 = np.prod(envelope1[:, 1] - envelope1[:, 0])
#         return -area_intersection  # 负面积用于最大化
#
#     # 初始移动值为0
#     initial_shift = np.array([0, 0])
#
#     # 最大化重叠面积
#     result = minimize(overlap_area, initial_shift, method='Nelder-Mead')
#
#     # 返回最佳移动值
#     best_shift = result.x
#     return best_shift
#
#
# # 示例坐标数组
# array1 = np.array([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])
# array2 = np.array([(3, 4), (4, 5), (5, 6), (6, 7), (7, 8)])
#
# best_shift = compute_overlap(array1, array2)
# print("最佳移动值 (dx, dy):", best_shift)
