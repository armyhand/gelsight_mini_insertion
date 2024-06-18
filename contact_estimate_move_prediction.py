import cv2
import numpy as np
from contact_detection import calibration
import math
from image_flow import Flow
from image_boundary import boundary_depth, choose_boundary
import time

"""
23.12.5 目标是实现夹取部件相对于夹爪的相对运动趋势检测，以及接触状态的判定和运动策略的制定
目标是实现轮廓特征点的选取，目前对于边缘点的提取较好，但是利用（如绘制边缘轮廓）效果不好，改成使用椭圆轮廓，但是会收到散点的影响(√)
使用例程解决了上述问题，例程给出的解决方法是选择最大的轮廓进行椭圆的拟合。接下来需要把长轴矢量提取出来，然后与运动趋势矢量进行结合(√)
根据通量和矢量夹角的关系判断运动趋势，目前的方案是根据轮廓椭圆的中心和长轴确定选取的点的范围，然后提取范围内点的光流矢量，根据椭圆短轴的方向把这些光流矢量
分为上、下两部分，分别计算上下部分和合力矢量与长轴矢量的夹角，然后根据夹角的大小判断相对运动趋势。（√）
把相对运动趋势表示出来，且尽可能减小误判（√）
存在的问题有左、右运动与左旋、右旋图像上显示相似：可以通过两指传感器互相印证（√）
把相对运动模态的提取看作一个模糊分类问题，可以实现光流运动趋势到相对运动趋势的映射；
从一指信息扩展到二指信息相互映射；从全局信息细化到局部信息
"""
# def extract_points_in_circle(flow, center, radius, angle):
#     # 创建一个与图像大小相同的网格，包含所有点的坐标
#     rows, cols = flow.shape[:2]
#     y, x = np.indices((rows, cols))
#
#     # 计算每个点到圆心的距离
#     distances = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
#
#     # 找到在给定半径内的点的索引
#     indices = np.where(distances <= radius)
#     # 提取位于圆内的点的坐标和像素值
#     # points = np.column_stack((indices[1], indices[0]))
#     force = flow[indices]
#
#     # 计算点相对于直线的位置
#     line_y = math.tan(math.radians(angle)) * (indices[1] - center[0]) + center[1]  # 直线方程，注意此时的图片坐标轴方向
#     above_line = indices[0] < line_y  # 点是否在直线上部分(这里越往上y坐标越小)
#
#     # 根据点的位置分别提取上部分和下部分的坐标和像素值
#     # points_above = points[above_line]
#     force_above = force[above_line]
#     # points_below = points[~above_line]
#     force_below = force[~above_line]
#
#     return force_above, force_below
#
# def force_calculation(flow):
#     flow_above = flow[0]
#     flow_below = flow[1]
#     Force_x_above = flow_above[:, 0]
#     Force_y_above = flow_above[:, 1]
#     Force_x_below = flow_below[:, 0]
#     Force_y_below = flow_below[:, 1]
#     force_x_above = np.mean(Force_x_above)
#     force_y_above = np.mean(Force_y_above)
#     force_x_below = np.mean(Force_x_below)
#     force_y_below = np.mean(Force_y_below)
#     # 计算光流合力
#     force_x = (force_x_above + force_x_below) / 2
#     force_y = (force_y_above + force_y_below) / 2
#     Force_above = np.array([force_x_above, force_y_above])
#     Force_below = np.array([force_x_below, force_y_below])
#     Force = np.array([force_x, force_y])
#     # 计算光流分布的夹角
#     angles_above = np.arctan2(flow_above[..., 1], flow_above[..., 0])
#     angles_above = np.degrees(angles_above)
#     angles_below = np.arctan2(flow_below[..., 1], flow_below[..., 0])
#     angles_below = np.degrees(angles_below)
#     angles = np.append(angles_above, angles_below)
#     angles_above_std = np.std(angles_above, axis=0)
#     angles_below_std = np.std(angles_below, axis=0)
#     angles_std = np.std(angles, axis=0) # 其大小可用于判断是否沿切向旋转
#
#     return Force, Force_above, Force_below, angles_std
#
# def Flux_calculation(axis, center, Force, Force_above, Force_below): #还需要更改成轮廓范围内的光流(√)
#     # Torque = np.zeros(0)  # the center when calculating the torque should be reset at the center of the object.
#     # for i in range(len(points)):
#     #     Torque = np.append(Torque, force[i][1] * (points[i, 0] - center[0]) + force[i, 0] * (points[i, 1] - center[1]))
#     theta = np.arccos(np.dot(axis, Force) / (np.linalg.norm(axis) * np.linalg.norm(Force)))  # 求解新老方向向量夹角
#     theta = math.degrees(theta)
#     theta_above = np.arccos(np.dot(axis, Force_above) / (np.linalg.norm(axis) * np.linalg.norm(Force_above)))  # 求解(上部分合力)新老方向向量夹角
#     theta_above = math.degrees(theta_above)
#     theta_below = np.arccos(np.dot(axis, Force_below) / (np.linalg.norm(axis) * np.linalg.norm(Force_below)))  # 求解(下部分合力)新老方向向量夹角
#     theta_below = math.degrees(theta_below)
#
#     # 判断旋转方向
#     cross_product = np.cross(axis, Force)
#     if cross_product < 0:
#         theta = -theta
#     cross_product = np.cross(axis, Force_above)
#     if cross_product < 0:
#         theta_above = -theta_above
#     cross_product = np.cross(axis, Force_below)
#     if cross_product < 0:
#         theta_below = -theta_below
#
#     return theta, theta_above, theta_below
#
# 读取GelSight图像
# 第一个传感器
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(
      "./force meaurment/piece_2_object/flow_init picture/flow_init.avi")  # choose to read from video or camera

# 第二个传感器
# cap_2 = cv2.VideoCapture(2)
cap_2 = cv2.VideoCapture('./force meaurment/piece_2_object/flow_init picture/flow_init_2.avi')
reference = cv2.imread(r'first time/flow_init2.png', cv2.IMREAD_COLOR)
ret, img = cap.read()

#绘制接触区域轮廓准备工作
cali = calibration()
pad = 20
ref_img = cv2.imread('test_data_2/ref.jpg')
ref_img = ref_img[cali.x_index, cali.y_index, :]
ref_img = cali.crop_image(ref_img, pad)
marker = cali.mask_marker(ref_img)
keypoints = cali.find_dots(marker)
marker_mask = cali.make_mask(ref_img, keypoints)
ref_img = cv2.inpaint(ref_img, marker_mask, 3, cv2.INPAINT_TELEA)
has_marke = True
k_roi = 0

# 绘制光流图准备工作
Flow = Flow(col=320, row=240)
col = 320
row = 240

time.sleep(1)
# 第一个传感器
ret, frame1 = cap.read()
# frame1 = Flow.get_raw_img(frame1)
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
f0 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
# 第二个传感器
ret, frame1_2 = cap_2.read()
# frame1_2 = Flow.get_raw_img(frame1_2)
prvs_2 = cv2.cvtColor(frame1_2, cv2.COLOR_BGR2GRAY)
f0_2 = cv2.cvtColor(frame1_2, cv2.COLOR_BGR2GRAY)

flow_left, flow_right = np.zeros((Flow.row, Flow.col, 2)), np.zeros((Flow.row, Flow.col, 2))
count = 0

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

# 计算通量准备工作
sum = np.zeros(0)

# 第一个传感器
ret_1, frame2_1 = cap.read()
# frame2_1 = Flow.get_raw_img(frame2_1) #连接传感器时使用
# 第二个传感器
ret_2, frame2_2 = cap_2.read()
# frame2_2 = Flow.get_raw_img(frame2_2) #连接传感器时使用
while ret_1 & ret_2 and cv2.waitKey(1) == -1:
    # 绘制接触区域轮廓
    # # img = cali.get_raw_img(img, 640, 480) #连接传感器时使用
    # img = cv2.resize(img, (640, 480))
    # img_boundary = img[cali.x_index, cali.y_index, :]
    # img_boundary = cali.crop_image(img_boundary, pad)
    # if has_marke:
    #     marker = cali.mask_marker(img_boundary)
    #     keypoints = cali.find_dots(marker)
    #     marker_mask = cali.make_mask(img_boundary, keypoints)
    # else:
    #     marker_mask = np.zeros_like(img_boundary)
    # contours = cali.contact_detection(img_boundary, ref_img, marker_mask)
    # if contours == None:
    #     cv2.imshow("contact", img)
    # else:
    #     count += 1
    #     im2show = cv2.ellipse(img, contours, (0, 255, 0), 2)
    #     # 提取椭圆主轴信息
    #     center, axis_lengths, angle = contours
    #     if abs(angle) > 90:
    #         angle = angle - 180  # 还需分类讨论
    #     major_axis_length = max(axis_lengths)
    #     major_axis = (int(center[0]),
    #                   int(center[1])), (
    #                      int(major_axis_length / 2 * math.sin(math.radians(angle)) + center[0]),
    #                      int(center[1] - major_axis_length / 2 * math.cos(math.radians(angle))))
    #     # 绘制主轴
    #     cv2.line(img, major_axis[0], major_axis[1], (0, 255, 0), 2)
    #     cv2.imshow('contact', im2show)

    # 绘制接触区域运动趋势光流图
    flow_left, flow_right, frame3_1, frame3_2, next_1, next_2 = Flow.cal_flow(f0, f0_2, frame2_1, frame2_2)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # K=ESC
        break

    ## 感知运动趋势在夹爪坐标系下的方向
    # 使用解析方法计算夹爪末端受力
    F_x, F_y, F_z, T_z = Flow.cal_gripper_force(flow_left, flow_right, delta=1)
    Force_x = np.append(Force_x, F_x)
    Force_y = np.append(Force_y, F_y)
    Force_z = np.append(Force_z, F_z)
    Torque_z = np.append(Torque_z, T_z)
    # 根据末端受力判断受力位置
    Force_threshold_x
    # 判断此时物体与孔的相对位置

    # ## 估计旋转中心点，提取ROI
    # if k_roi == 0:
    #     r0 = np.array([img.shape[0], img.shape[1]]) / 2 # 图像的中心点
    #     # 获取按照元素值大小排序后的索引
    #     sorted_indices = np.argsort(flow_sum, axis=None)
    #     # 获取排序后的坐标位置
    #     sorted_coordinates = np.column_stack(np.unravel_index(sorted_indices, flow_sum.shape))
    #     # 指定坐标范围
    #     xmin, xmax = r0[0]-50, r0[0]+50
    #     ymin, ymax = r0[1]-50, r0[1]+50
    #     # 根据坐标范围筛选坐标位置
    #     filtered_coordinates = [coord for coord in sorted_coordinates if
    #                             xmin <= coord[0] <= xmax and ymin <= coord[1] <= ymax]
    #     cv2.drawMarker(frame2, filtered_coordinates[-1][:2], color=(0,0,255), markerSize=4, thickness=2)

    #
#     # 计算通量的大小
#     if count < 10:
#         sum = np.append(sum, angle)
#         print('initialization...')
#     if count == 10:
#         Angle = np.mean(sum) #得到工件方向角度
#         axis = np.array([1.0 * math.sin(math.radians(Angle)), -1.0 * math.cos(math.radians(Angle))]) #工件方向矢量
#     if count > 10:
#         force_above, force_below = extract_points_in_circle(flow_sum, center, major_axis_length / 2, Angle) # 提取出在圆内（后续考虑更改成椭圆内）的光流并分成上下两部分
#         flow_in = [force_above, force_below]
#         Flow_all, Flow_above, Flow_below, Angles_std = force_calculation(flow_in)
#         theta, theta_above, theta_below = Flux_calculation(axis, center, Flow_all, Flow_above, Flow_below)
#         Force = math.sqrt((Flow_all[0])**2 + (Flow_all[1])**2)
#         print('theta=%f, theta_above=%f, theta_below=%f, Force=%f, Angles_std=%f'%(theta, theta_above, theta_below, Force, Angles_std))
#         if contours != None:
#             flow_x_above = Flow_above[0]
#             flow_y_above = Flow_above[1]
#             Flow.draw_sumLine(frame2, (
#                          int(major_axis_length / 4 * math.sin(math.radians(angle)) + center[0]),
#                          int(center[1] - major_axis_length / 4 * math.cos(math.radians(angle)))), flow_x_above, flow_y_above)
#             flow_x_below = Flow_below[0]
#             flow_y_below = Flow_below[1]
#             Flow.draw_sumLine(frame2, (
#                          int(center[0] - major_axis_length / 4 * math.sin(math.radians(angle))),
#                          int(center[1] + major_axis_length / 4 * math.cos(math.radians(angle)))), flow_x_below, flow_y_below)
#             flow_x = Flow_all[0]
#             flow_y = Flow_all[1]
#             Flow.draw_sumLine(frame2, center, flow_x, flow_y)
#     cv2.imshow('frame', frame2)
#
#     if cv2.waitKey(40) & 0xFF == ord('q'):
#         break
#
    # 第一个传感器
    ret_1, frame2_1 = cap.read()
    # frame2_1 = Flow.get_raw_img(frame2_1) #连接传感器时使用
    # 第二个传感器
    ret_2, frame2_2 = cap_2.read()
    # frame2_2 = Flow.get_raw_img(frame2_2) #连接传感器时使用

cv2.destroyAllWindows()
cap.release()
