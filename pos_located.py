#!/usr/local/bin/python
# encoding: utf-8
# File Name: optical_flow.py
# Author: Shaoxiong (Shawn) Wang
# Create Time: 2017/11/15 16:24
# TODO:
# Add annotation
import cv2
import numpy as np
import numpy.matlib
import time
from gelsight import gsdevice
import matplotlib.pyplot as plt
import socket
import struct
from ARIE_range import beta_to_xy, xy_to_beta

"""
光流法显示标记点的运动趋势
检测实际接触位置与接触线是否平行并使用TCP通信传输调整值
"""


class Flow:
    def __init__(self, col, row):
        x0 = np.matlib.repmat(np.arange(row), col, 1).T
        y0 = np.matlib.repmat(np.arange(col), row, 1)

        self.x = np.zeros_like(x0).astype(int)
        self.y = np.zeros_like(y0).astype(int)
        self.x0 = x0
        self.y0 = y0
        self.col = col
        self.row = row

        self.out = cv2.VideoWriter('flow picture/flow.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3.5,
                                   (col * 1, row * 1))  # The fps depends on CPU
        self.out_2 = cv2.VideoWriter('flow_init picture/flow_init.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3.5,
                                     (col * 1, row * 1))
        self.out_3 = cv2.VideoWriter('flow picture/flow_2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3.5,
                                     (col * 1, row * 1))  # The fps depends on CPU
        self.out_4 = cv2.VideoWriter('flow_init picture/flow_init_2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                     3.5, (col * 1, row * 1))

        self.t1 = time.time()
        self.t2 = time.time()

    def get_raw_img(self, frame):
        img = cv2.resize(frame, (895, 672))  # size suggested by janos to maintain aspect ratio
        border_size_x, border_size_y = int(img.shape[0] * (1 / 7)), int(
            np.floor(img.shape[1] * (1 / 7)))  # remove 1/7th of border from each size
        img = img[border_size_x:img.shape[0] - border_size_x, border_size_y:img.shape[1] - border_size_y]
        img = img[:, :-1]  # remove last column to get a popular image resolution
        img = cv2.resize(img, (self.col, self.row))  # final resize for 3d

        return img

    def add_flow(self, flow):
        dx = np.round(self.x + self.x0).astype(int)
        dy = np.round(self.y + self.y0).astype(int)
        dx[dx >= self.row] = self.row - 1
        dx[dx < 0] = 0
        dy[dy >= self.col] = self.col - 1
        dy[dy < 0] = 0
        ds = np.reshape(flow[np.reshape(dx, -1), np.reshape(dy, -1)], (self.row, self.col, -1))
        nx = self.x + ds[:, :, 0]
        ny = self.y + ds[:, :, 1]
        return nx, ny

    def flow2color(self, flow, hsv, K=15):  # make the flow vector into the rgb color of each pixel
        mag, ang = cv2.cartToPolar(-flow[..., 1], flow[..., 0])
        hsv[..., 0] = ang * 180 / np.pi / 2
        mag = mag.astype(float) * K * 960 / self.col

        mag[mag > 255] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = mag
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return bgr

    def draw(self, img, flow, scale=2.0):  # draw the arrowedline in the image
        start_line = 10
        start_vertical = 10
        step = 10
        d_all = np.zeros((round((self.row - start_vertical) / step), round((self.col - start_line) / step), 2))
        m, n = 0, 0
        for i in range(start_vertical, self.row, step):
            for j in range(start_line, self.col, step):
                d = (flow[i, j] * scale).astype(int)
                cv2.arrowedLine(img, (j, i), (j + d[0], i + d[1]), (0, 255, 255),
                                1)  # cv2.arrowedLine(img, startpoint(x,y), endpoint(x,y), color, linedwidth)
                d_all[m, n] = d / scale
                n += 1
            m += 1
            n = 0

        return d_all

    def cal_force(self, flow,
                  delta):  # claculate the trendlines' vector sum to represent the force in the paralleral and the torque of the z axis.
        force_x = 0
        force_y = 0
        torque = 0  # the center when calculating the torque should be reset at the center of the object.
        height = self.row
        width = self.col
        force_x = np.mean(flow[:, :, 0])
        force_y = np.mean(flow[:, :, 1])
        torque = np.mean(np.gradient(flow[:, :, 1])[1] - np.gradient(flow[:, :, 0])[0]) / delta

        return force_x, force_y, torque

    def draw_sumLine(self, img, center, sum_x, sum_y, scale=5.0):
        height = img.shape[0]
        width = img.shape[1]
        cv2.arrowedLine(img, (int(center[0]), int(center[1])),
                        (int(center[0] + sum_x * scale), int(center[1] + sum_y * scale)),
                        (0, 0, 255), 2)

    def heatmap(self, flow, min_flow, max_flow):
        Flow_min = np.ones(self.col) * min_flow
        Flow_max = np.ones(self.col) * max_flow
        # 计算光流大小（模长）
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        magnitude = np.append(magnitude, Flow_min).reshape(-1, self.col)
        magnitude = np.append(magnitude, Flow_max).reshape(-1, self.col)

        # 将光流大小映射到热力图颜色范围
        heatmap = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # 归一化
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = heatmap[0:self.row]

        return heatmap

    def cal_div_curl(self, v_field, delta):
        div = np.mean(np.gradient(v_field[:, :, 0])[1] + np.gradient(v_field[:, :, 1])[0]) / delta
        curl = np.mean(np.gradient(v_field[:, :, 1])[1] - np.gradient(v_field[:, :, 0])[0]) / delta

        return div, curl

    def cal_gripper_force(self, flow_left, flow_right, delta):
        # 左侧传感器受力情况
        force_x_left, force_y_left, torque_left = self.cal_force(flow_left, delta)
        div_left, curl_left = self.cal_div_curl(flow_left, delta)
        # 右侧传感器受力情况
        force_x_right, force_y_right, torque_right = self.cal_force(flow_right, delta)
        div_right, curl_right = self.cal_div_curl(flow_right, delta)
        # 计算夹爪受力情况
        F_x = -(force_x_left + force_x_right) * 1
        F_y = (curl_left - curl_right) / 2
        F_z = (force_x_left - force_x_right) / 2
        T_z = (force_y_left - force_y_right) * 1

        return F_x, F_y, F_z, T_z

    def cal_flow(self, f0, f0_2, frame2_1, frame2_2):
        next_1 = cv2.cvtColor(frame2_1, cv2.COLOR_BGR2GRAY)
        next_2 = cv2.cvtColor(frame2_2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(f0, next_1, None, 0.5, 3, int(180 * self.col / 960), 5, 5, 1.2, 0)
        flow_left = flow
        self.t1 = time.time()

        frame3_1 = np.copy(frame2_1)
        d_flow = self.draw(frame3_1, flow_left)  # frame3是此刻与初始时刻的光流图（校正后）
        cv2.imshow('frame', frame3_1)

        self.out.write(frame3_1)
        self.out_2.write(frame2_1)

        # 第二个传感器
        flow = cv2.calcOpticalFlowFarneback(f0_2, next_2, None, 0.5, 3, int(180 * self.col / 960), 5, 5, 1.2, 0)
        flow_right = flow
        self.t2 = time.time()

        frame3_2 = np.copy(frame2_2)
        d_flow = self.draw(frame3_2, flow_right)  # frame3是此刻与初始时刻的光流图（校正后）

        cv2.imshow('frame2', frame3_2)

        self.out_3.write(frame3_2)
        self.out_4.write(frame2_2)

        return flow_left, flow_right, frame3_1, frame3_2, next_1, next_2


if __name__ == "__main__":
    # 第一个传感器
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(
    #     "peg_hole_insertion_test/star_hole_x/star_insertion_7.5degree_2/flow_init picture/flow_init.avi")  # choose to read from video or camera

    # 第二个传感器
    cap_2 = cv2.VideoCapture(2)
    # cap_2 = cv2.VideoCapture('peg_hole_insertion_test/star_hole_x/star_insertion_7.5degree_2/flow_init picture/flow_init_2.avi')

    Flow = Flow(col=320, row=240)
    col = 320
    row = 240

    time.sleep(1)
    # 第一个传感器
    ret, frame1 = cap.read()
    frame1 = Flow.get_raw_img(frame1)
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    f0 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # 第二个传感器
    ret, frame1_2 = cap_2.read()
    frame1_2 = Flow.get_raw_img(frame1_2)
    prvs_2 = cv2.cvtColor(frame1_2, cv2.COLOR_BGR2GRAY)
    f0_2 = cv2.cvtColor(frame1_2, cv2.COLOR_BGR2GRAY)

    flow_left, flow_right = np.zeros((Flow.row, Flow.col, 2)), np.zeros((Flow.row, Flow.col, 2))

    min_flow = 0  # 光流的最小值
    max_flow = 15  # 光流的最大值

    reset_threshold_error = 0.3  # ?
    reset_threshold_mean = 2  # ?
    Div = np.zeros(0)
    Curl = np.zeros(0)
    Force_x_left = np.zeros(0)
    Force_y_left = np.zeros(0)
    Torque_left = np.zeros(0)
    Force_x_right = np.zeros(0)
    Force_y_right = np.zeros(0)
    Torque_right = np.zeros(0)
    Force_x = np.zeros(0)
    Force_y = np.zeros(0)
    Force_z = np.zeros(0)
    Torque_z = np.zeros(0)

    # 定义一个ip协议版本AF_INET，为IPv4；同时也定义一个传输协议（TCP）SOCK_STREAM
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 定义IP地址与端口号
    ip_port = ('172.31.1.147', 30001)
    # 进行连接服务器
    client.connect(ip_port)
    A = []
    B = []
    Frq = []
    # 比例控制进行调整,相关参数
    keep = 0
    delta_x = 0 #绕x轴旋转的度数
    delta_y = 0 #绕y轴旋转的度数
    delta_z = 0
    beta = 0
    Kx = 5
    Kz = 1
    Force_x_keep = np.zeros(0)  # step=0且稳定接触平面之后的x方向运动趋势集合
    Force_y_keep = np.zeros(0)  # step=0且稳定接触平面之后的y方向运动趋势集合
    Force_z_keep = np.zeros(0)
    Torque_z_keep = np.zeros(0)
    Force_x_rotate = np.zeros(0)
    Force_y_rotate = np.zeros(0)
    Force_z_rotate = np.zeros(0)
    Torque_z_rotate = np.zeros(0)
    F_x_keep = 0  # x方向运动趋势的平均值
    F_y_keep = 0  # y方向运动趋势的平均值
    F_z_keep = 0
    R_z_keep = 0
    F_x_rotate = 0
    F_y_rotate = 0
    F_z_rotate = 0
    R_z_rotate = 0
    X0 = 0
    F_rotate = []
    flag = 1 #标志，为1代表可以写入，为0代表不能写入
    flash = 0
    count_n = 0
    Threshold = False

    while (1):
        # count += 1
        try:
            # 第一个传感器
            ret_1, frame2_1 = cap.read()
            frame2_1 = Flow.get_raw_img(frame2_1) #连接传感器时使用
            next_1 = cv2.cvtColor(frame2_1, cv2.COLOR_BGR2GRAY)
            # 第二个传感器
            ret_2, frame2_2 = cap_2.read()
            frame2_2 = Flow.get_raw_img(frame2_2) #连接传感器时使用
            next_2 = cv2.cvtColor(frame2_2, cv2.COLOR_BGR2GRAY)
        except:
            break
        flow_left, flow_right, frame3_1, frame3_2, next_1, next_2 = Flow.cal_flow(f0, f0_2, frame2_1, frame2_2)
        # # 第一个传感器
        # try:
        #     ret_1, frame2_1 = cap.read()
        #     # frame2_1 = Flow.get_raw_img(frame2_1)
        #     next_1 = cv2.cvtColor(frame2_1, cv2.COLOR_BGR2GRAY)
        #     t1 = time.time()
        # except:
        #     break
        #
        # flow = cv2.calcOpticalFlowFarneback(f0, next_1, None, 0.5, 3, int(180 * Flow.col / 960), 5, 5, 1.2, 0)
        # flow_left = flow
        #
        # frame3_1 = np.copy(frame2_1)
        # d_flow = Flow.draw(frame3_1, flow_left)  # frame3是此刻与初始时刻的光流图（校正后）
        # cv2.imshow('frame', frame3_1)
        #
        # k = cv2.waitKey(30) & 0xff
        # Flow.out.write(frame3_1)
        # Flow.out_2.write(frame2_1)
        # # cv2.imwrite(r'flow_init picture/flow_init' + str(count) + '.png', frame2)
        # # cv2.imwrite(r'flow picture/flow' + str(count) + '.png', frame3)
        # if k == 27:  # K=ESC
        #     break
        # prvs = next_1  #
        #
        # # 第二个传感器
        # try:
        #     ret_2, frame2_2 = cap_2.read()
        #     # frame2_2 = Flow.get_raw_img(frame2_2)
        #     next_2 = cv2.cvtColor(frame2_2, cv2.COLOR_BGR2GRAY)
        # except:
        #     break
        #
        # flow = cv2.calcOpticalFlowFarneback(f0_2, next_2, None, 0.5, 3, int(180 * Flow.col / 960), 5, 5, 1.2, 0)
        # flow_right = flow
        #
        # frame3_2 = np.copy(frame2_2)
        # d_flow_2 = Flow.draw(frame3_2, flow_right)  # frame3是此刻与初始时刻的光流图（校正后）
        #
        # cv2.imshow('frame2', frame3_2)
        #
        # Flow.out_3.write(frame3_2)
        # Flow.out_4.write(frame2_2)
        # prvs_2 = next_2  #

        k = cv2.waitKey(30) & 0xff
        if k == 27:  # K=ESC
            break

        frq = Flow.t2 - Flow.t1
        Frq.append(frq)
        print("frequency= ", frq)

        # 显示散度和旋度
        div_left, curl_left = Flow.cal_div_curl(flow_left, delta=1)
        flow_x_left, flow_y_left, flow_t_left = Flow.cal_force(flow_left, delta=1)
        # print("frame_left :div= %.3f, curl= %.3f, flow_x= %.3f, flow_y= %.3f, torque= %.3f" % (
        #     div_left, curl_left, flow_x_left, flow_y_left, flow_t_left))
        Force_x_left = np.append(Force_x_left, flow_x_left)
        Force_y_left = np.append(Force_y_left, flow_y_left)
        Torque_left = np.append(Torque_left, flow_t_left)
        # 显示散度和旋度
        div_right, curl_right = Flow.cal_div_curl(flow_right, delta=1)
        flow_x_right, flow_y_right, flow_t_right = Flow.cal_force(flow_right, delta=1)
        # print("frame_right:div= %.3f, curl= %.3f, flow_x= %.3f, flow_y= %.3f, torque= %.3f" % (
        #     div_right, curl_right, flow_x_right, flow_y_right, flow_t_right))
        Force_x_right = np.append(Force_x_right, flow_x_right)
        Force_y_right = np.append(Force_y_right, flow_y_right)
        Torque_right = np.append(Torque_right, flow_t_right)

        # 使用解析方法计算夹爪末端受力
        F_x, F_y, F_z, T_z = Flow.cal_gripper_force(flow_left, flow_right, delta=1)
        Force_x = np.append(Force_x, F_x)
        Force_y = np.append(Force_y, F_y)
        Force_z = np.append(Force_z, F_z)
        Torque_z = np.append(Torque_z, T_z)
        print("frame_force:T_x= %.3f, T_y= %.3f, T_z= %.3f, R_z= %.3f" % (F_x, F_y, F_z, T_z))
        message = [keep, delta_x, delta_y, delta_z, flash]

        message_str = 'r' + str(message) + '\n'
        client.sendall(message_str.encode('utf-8'))
        received_data = client.recv(1024)  # 接受服务端的信息，最大数据为1k
        floats = []
        if len(received_data) == 144:
            floats_bytes = received_data
            B.append(received_data)
            float_count = len(floats_bytes) // 4
            floats = list(struct.unpack('>' + 'f' * float_count, floats_bytes))
            A.append(floats)
            print("received_data= ",floats)
        if received_data[0] == 1 and len(received_data) > 144:
            start_idx = received_data.find(b'\x01')  # 找到起始标志的索引
            if len(received_data) == 145:
                floats_bytes = received_data[start_idx + 1:]  # 提取两个标志之间的数据
            elif received_data[145] == 1:
                end_idx = 145  # 找到结束标志的索引
                floats_bytes = received_data[start_idx + 1:end_idx]  # 提取两个标志之间的数据
            else:
                continue
            B.append(received_data)
            float_count = len(floats_bytes) // 4
            floats = list(struct.unpack('>' + 'f' * float_count, floats_bytes))
            A.append(floats)
            print("received_data= ", floats)
        print("------------------------------------------------------------")

        ##绕Z轴矩感知-最小化
        if len(floats) > 0:
            step = floats[14]
            print("step=", step)
            if step == 1 and F_z >= 4: #稳定接触的z运动趋势大小可以调整
                vertical_constrain = floats[13]
                F_x_keep = 0
                F_y_keep = 0
                F_z_keep = 0
                R_z_keep = 0
                F_x_rotate = 0
                F_y_rotate = 0
                F_z_rotate = 0
                R_z_rotate = 0
                if flag == 1:  # 数组已被清零，可以填充数据
                    Force_x_keep = np.append(Force_x_keep, F_x)
                    Force_y_keep = np.append(Force_y_keep, F_y)
                    Force_z_keep = np.append(Force_z_keep, F_z)
                    Torque_z_keep = np.append(Torque_z_keep, T_z)
                else:
                    print("step=0 error")
                    break
                if vertical_constrain == 1:
                    X0 = floats[21]
                    Y0 = floats[22]
                keep = 1
                delta_x = 0
                delta_z = 0
                flash = 0

            if step == 2:
                keep = 0
                delta_x = 0
                delta_z = 0
                flash = 0
                vertical_constrain = floats[13]
                F_x_keep = np.mean(Force_x_keep[1:len(Force_x_keep) - 1])
                F_y_keep = np.mean(Force_y_keep[1:len(Force_y_keep) - 1])
                F_z_keep = np.mean(Force_z_keep[1:len(Force_z_keep) - 1])
                R_z_keep = np.mean(Torque_z_keep[1:len(Force_y_keep) - 1])
                if vertical_constrain == 1:
                    X1 = floats[21]
                    Y1 = floats[22]

            # if step == 3 and X0 != 0:
            #     keep = 0
            #     end1 = floats[17]
            #     end2 = floats[18]
            #     vertical_constrain = floats[13]
            #     X1 = floats[21]
            #     Y1 = floats[22]
            #
            #     if flash == 1 and end1 == 0:
            #         flag = 1
            #         flash = 0
            #         F_x_rotate = 0
            #         F_y_rotate = 0
            #         F_z_rotate = 0
            #         R_z_rotate = 0
            #
            #     if end1 == 1 and flag == 1: #计算旋转趋势的平均值
            #         Force_x_rotate = np.append(Force_x_rotate, F_x)
            #         Force_y_rotate = np.append(Force_y_rotate, F_y)
            #         Force_z_rotate = np.append(Force_z_rotate, F_z)
            #         Torque_z_rotate = np.append(Torque_z_rotate, T_z)
            #     if end2 == 1 and len(Force_x_rotate) > 7 and flag == 1:
            #         flag = 0
            #         F_x_rotate = np.mean(Force_x_rotate[1:len(Force_x_rotate) - 1])
            #         F_y_rotate = np.mean(Force_y_rotate[1:len(Force_y_rotate) - 1])
            #         F_z_rotate = np.mean(Force_z_rotate[1:len(Force_z_rotate) - 1])
            #         R_z_rotate = np.mean(Torque_z_rotate[1:len(Torque_z_rotate) - 1])
            #         if abs(X0 - X1) <= 0.5: #当Rz小于阈值时，不再进一步调整
            #             Threshold = True
            #
            #     if F_z_rotate != 0 and Threshold == False:
            #         Force_x_rotate = np.zeros(0)
            #         Force_y_rotate = np.zeros(0)
            #         Force_z_rotate = np.zeros(0)
            #         Torque_z_rotate = np.zeros(0)
            #         F_rotate.append([F_x_rotate, F_y_rotate, F_z_rotate, R_z_rotate])
            #         keep = 0
            #         delta_x = X0 - X1
            #         delta_y = Y0 - Y1
            #         delta_z = -Kz * (6 - F_z_rotate) #恒力为6N
            #         flash = 1
            #         count = floats[19]  # 临时这样，后面删去此段代码
            #         if count >= 10:
            #             Threshold = True
            #     elif Threshold == True:
            #         keep = 1
            #         delta_x = 0
            #         delta_y = 0
            #         delta_z = 0
            #         flash = 0
            #         F_rotate.append([F_x_rotate, F_y_rotate, F_z_rotate, R_z_rotate])
            #     else:
            #         keep = 0
            #         delta_x = 0
            #         delta_y = 0
            #         delta_z = 0
            #         flash = 0

            # if step == 3 and X0 != 0:
            #     keep = 0
            #     end1 = floats[17]
            #     end2 = floats[18]
            #     vertical_constrain = floats[13]
            #     X1 = floats[21]
            #     Y1 = floats[22]
            #
            #     if end1 == 1 and flag == 1: #计算旋转趋势的平均值
            #         Force_x_rotate = np.append(Force_x_rotate, F_x)
            #         Force_y_rotate = np.append(Force_y_rotate, F_y)
            #         Force_z_rotate = np.append(Force_z_rotate, F_z)
            #         Torque_z_rotate = np.append(Torque_z_rotate, T_z)
            #     if end2 == 1 and len(Force_x_rotate) > 7 and flag == 1:
            #         flag = 0
            #         F_x_rotate = np.mean(Force_x_rotate[1:len(Force_x_rotate) - 1])
            #         F_y_rotate = np.mean(Force_y_rotate[1:len(Force_y_rotate) - 1])
            #         F_z_rotate = np.mean(Force_z_rotate[1:len(Force_z_rotate) - 1])
            #         R_z_rotate = np.mean(Torque_z_rotate[1:len(Torque_z_rotate) - 1])
            #
            #
            #     if Threshold == False:
            #         keep = 0
            #         delta_x = np.sign(-T_z)
            #         delta_y = np.sign(-T_z)
            #         delta_z = -Kz * (6 - F_z) #恒力为6N
            #         if abs(T_z) <= 0.1 and count_n >= 5:
            #             Threshold = True
            #         if abs(T_z) <= 0.1:
            #             count_n += 1
            #         if abs(T_z) >= 0.2:
            #             count_n = 0
            #     else:
            #         if end1 == 0:
            #             flash = 1 # 客户端收到后把end1更改成1
            #             keep = 0
            #             delta_x = 0
            #             delta_y = 0
            #             delta_z = 0
            #         if len(Force_x_rotate) > 10:
            #             flash = 2 #客户端收到后把end2改成1
            #             keep = 0
            #             delta_x = 0
            #             delta_y = 0
            #             delta_z = 0
            #         if end2 == 1 and F_x_rotate != 0:
            #             keep = 1
            #             delta_x = 0
            #             delta_y = 0
            #             delta_z = 0

        else:
            keep = 0
            delta_x = 0
            delta_y = 0
            delta_z = 0
            flash = 0


        if k == ord("r"):  # to make a reset of the flow
            f0 = next_1
            f0_2 = next_2

    cap.release()
    cv2.destroyAllWindows()
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(len(Force_x_left)), Force_x_left, label='left')
    plt.plot(np.arange(len(Force_x_right)), Force_x_right, label='right')
    plt.ylabel('X')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(len(Force_y_left)), Force_y_left, label='left')
    plt.plot(np.arange(len(Force_y_right)), Force_y_right, label='right')
    plt.ylabel('Y')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(len(Torque_left)), Torque_left, label='left')
    plt.plot(np.arange(len(Torque_right)), Torque_right, label='right')
    plt.ylabel('Tz')
    plt.xlabel('pixel')
    plt.legend()
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(np.arange(len(Force_x)), Force_x, label='F_x')
    plt.legend()
    plt.subplot(4, 1, 2)
    plt.plot(np.arange(len(Force_y)), Force_y, label='F_y')
    plt.legend()
    plt.subplot(4, 1, 3)
    plt.plot(np.arange(len(Force_z)), Force_z, label='F_z')
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(np.arange(len(Torque_z)), Torque_z, label='T_z')
    plt.xlabel('pixel')
    plt.legend()
    client.close()
    if len(A) > 100:
        A1 = np.array(A)
        np.savetxt(r'force_bytes.txt', A)