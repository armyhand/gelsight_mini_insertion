import cv2
import numpy as np
from contact_detection import calibration
import math
from image_boundary import boundary_depth, choose_boundary
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def contour_involve(points, contour, dx, dy):
    x_min = contour[0]
    x_max = contour[0] + dx
    y_max = contour[1]
    y_min = contour[1] - dy
    indices = np.where((points[:,0] > x_min) & (points[:,0] < x_max) & (points[:,1] > y_min) & (points[:,1] < y_max))[0]
    if len(indices) > 1:
        return True
    else:
        return False

def get_distance_point2line(point, line):
    """
    Args:
        point: [x0, y0]
        line: [x1, y1, x2, y2]
    """
    line_point1, line_point2 = np.array(line[0:2]), np.array(line[2:])
    vec1 = line_point1 - point
    vec2 = line_point2 - point
    distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)

    return distance

def angel_calculation(axis, Force):
    """
    :param axis: [x, y]
    :param Force: [x, y]
    :return: degree
    """
    theta = np.arccos(np.dot(axis, Force) / (np.linalg.norm(axis) * np.linalg.norm(Force)))  # 求解新老方向向量夹角
    theta = math.degrees(theta)

    return theta

##定义一些全局变量
nn = boundary_depth(cam_id=0)
Calibration = True
col = 320
row = 240

##校准传感器
if  Calibration == None:
    file_path = r'wujiaoxing_pressure_meaurment/contour_points_2.txt'
    # 下载边缘像素点的坐标
    reference_pixel_n = np.loadtxt(file_path)
    plt.scatter(reference_pixel_n[:, 0], reference_pixel_n[:, 1])
    plt.show()
    plt.figure()

    object_left_pixel = input("零件图像最左端横坐标：")
    object_right_pixel = input("零件图像最右端横坐标：")
    object_bottom_pixel = input("零件图像最下端纵坐标：")
    object_top_pixel = input("零件图像最上端纵坐标：")
    object_width_real = input("零件的实际宽度（mm）：")
    object_height_real = input("零件的实际高度（mm）：")

    l_pixel = int(object_left_pixel)
    r_pixel  =int(object_right_pixel)
    b_pixel = int(object_bottom_pixel)
    t_pixel = int(object_top_pixel)
    w_real = int(object_width_real)
    h_real = int(object_height_real)

    dL = w_real * col / (r_pixel - l_pixel)
    dH = h_real * row / (t_pixel - b_pixel)
else:
    dL = 20.42  # 横坐标（宽度）对应的尺度变换 （mm)
    dH = 15.0  # 纵坐标（高度）对应的尺度变换（mm）

##确定初始扫掠范围
# Specify the path to your text file
file_path = r'object_coordinate/star_hole.txt'
# Initialize an empty list to store the (x, y, z) tuples
data_list = []
# Read the file line by line and extract (x, y, z) values
with open(file_path, 'r') as file:
    for line in file:
        # Remove parentheses and split by comma
        values = line.strip('()\n').split(',')
        # Convert values to float and append to the list
        x, y, z = map(float, values)
        data_list.append([x, y])
# Convert the list of lists to a Numpy array
data_array = np.array(data_list)
# 创建一个图形
fig, ax = plt.subplots()
ax.scatter(data_array[:, 0], data_array[:, 1])
plt.xticks(fontsize=18, fontproperties='Times New Roman')
plt.yticks(fontsize=18, fontproperties='Times New Roman')
plt.xlabel('X(mm)', fontsize=20, fontdict={'family': 'Times New Roman'})
plt.ylabel('Y(mm)', fontsize=20, fontdict={'family': 'Times New Roman'})
# plt.title('the origin vision contour')
plt.show()
# x, y, W, H = cv2.boundingRect(np.array(data_array, dtype=np.float32))
# 确定框出的边界
min_x = np.min(data_array[:,0])
max_x = np.max(data_array[:,0])
min_y = np.min(data_array[:,1])
max_y = np.max(data_array[:,1])
x = min_x - 20  # 左上角横坐标，20为裕度
y = max_y + 20  # 左上角纵坐标
W = max_x - min_x + 40
H = max_y - min_y + 40
# 显示框出后的区域
# 创建一个矩形对象
fig, ax = plt.subplots()
ax.scatter(data_array[:, 0], data_array[:, 1])
rect = patches.Rectangle((x, y-H), W, H, linewidth=1, edgecolor='r', facecolor='none')
# 添加矩形到图形
ax.add_patch(rect)
plt.xticks(fontsize=18, fontproperties='Times New Roman')
plt.yticks(fontsize=18, fontproperties='Times New Roman')
plt.xlabel('X(mm)', fontsize=20, fontdict={'family': 'Times New Roman'})
plt.ylabel('Y(mm)', fontsize=20, fontdict={'family': 'Times New Roman'})
plt.title('the origin vision contour and boundary')
plt.show()

##划分扫掠区域
h = 12  # 纵向接触有效区域(mm)
w = 14  # 横向接触有效区域(mm)
alpha = 0.1  # 相邻区域的重叠比率
m = (W - alpha * h) // ((1 - alpha) * h) + np.sign((W - alpha * h) % ((1 - alpha) * h))  # 横向覆盖矩形框所需有效接触区域个数（注意机械臂坐标系的横坐标对应传感器的纵向接触区域）
n = (H - alpha * w) // ((1 - alpha) * w) + np.sign((H - alpha * w) % ((1 - alpha) * w))  # 纵矩形框所需有效接触区域个数
cover = np.zeros((int(n), int(m), 2))  # 初始覆盖满最小矩形框所需的接触区域数目为：(n)*(m)个
cover[0, 0] = np.array([x, y])  # 左上角矩形为起点(选取的特征点为接触区域矩形的左上点)
for i in range(int(n)):
    for j in range(int(m)):
        cover[i, j] = np.array([x + j * (1 - alpha) * h, y - i * (1 - alpha) * w])  # cover保存的矩形坐标点为其左上角点
# 显示划分后的区域
fig, ax = plt.subplots()
ax.scatter(data_array[:, 0], data_array[:, 1])
for i in range(int(n)):
    for j in range(int(m)):
        rect = patches.Rectangle((cover[i, j, 0], cover[i, j, 1]-w), h, w, linewidth=1, edgecolor='r', facecolor='none')
        # 添加矩形到图形
        ax.add_patch(rect)
plt.xticks(fontsize=18, fontproperties='Times New Roman')
plt.yticks(fontsize=18, fontproperties='Times New Roman')
plt.xlabel('X(mm)', fontsize=20, fontdict={'family': 'Times New Roman'})
plt.ylabel('Y(mm)', fontsize=20, fontdict={'family': 'Times New Roman'})
plt.show()

##进行拓扑推理，减少扫描区域
data_array_mean = np.mean(data_array, axis=0)  # 求解区域的质心
scale_x = (max_x - min_x + 40) / (max_x - min_x)  # 缩放比例
scale_y = (max_y - min_y + 40) / (max_y - min_y)  # 缩放比例
data_array_outer = data_array
data_array_outer[:,0] = data_array_mean[0] + scale_x * (data_array[:,0] - data_array_mean[0])  # 得到最终的缩放后的轮廓点坐标
data_array_outer[:,1] = data_array_mean[1] + scale_y * (data_array[:,1] - data_array_mean[1])  # 得到最终的缩放后的轮廓点坐标
cover_reduce = [] # 把与缩放后轮廓点有交集的扫描区域提取出来
for i in range(int(n)):
    for j in range(int(m)):
        if contour_involve(data_array_outer, cover[i, j], h, w):
            cover_reduce.append(list(cover[i, j]))
cover_overlap = np.zeros((len(cover_reduce), len(cover_reduce), 5))  # 定义接触区域之间两两重合的区域，每个区域表示为（x,y,w,h, flag）五维形式
for i in range(len(cover_reduce) - 1):
    [x1, y1] = cover_reduce[i]
    for j in range(i + 1, len(cover_reduce)):
        [x2, y2] = cover_reduce[j]
        if max(x1, x2) - min(x1, x2) > h or max(y1, y2) - min(y1, y2) > w:
            cover_overlap[i, j] = np.array([0, 0, 0, 0, 0]) # 表示无相接的重合区域
            continue
        xt = max(x1, x2)
        yt = min(y1, y2)
        Wt = min(x1, x2) + h - xt
        Ht = yt + w - max(y1, y2)
        cover_overlap[i, j] = np.array([xt, yt, Wt, Ht, 1])  # 表示有相接的重合区域
# 显示减少后的划分区域
fig, ax = plt.subplots()
ax.scatter(data_array_outer[:, 0], data_array_outer[:, 1])
for i in range(len(cover_reduce)):
    rect = patches.Rectangle((cover_reduce[i][0], cover_reduce[i][1]-w), h, w, linewidth=1, edgecolor='r', facecolor='none')
    # 添加矩形到图形
    ax.add_patch(rect)
# rect = patches.Rectangle((x, y-H), W, H, linewidth=1, edgecolor='r', facecolor='none')
# ax.add_patch(rect)
plt.xticks(fontsize=18, fontproperties='Times New Roman')
plt.yticks(fontsize=18, fontproperties='Times New Roman')
plt.xlabel('X(mm)', fontsize=20, fontdict={'family': 'Times New Roman'})
plt.ylabel('Y(mm)', fontsize=20, fontdict={'family': 'Times New Roman'})
plt.show()

##确定扫掠区域的优先级排序
Cost = []
# 对各个扫掠区域进行打分
point_1 = data_array_mean
for i in range(len(cover_reduce)):
    [x0, y0] = cover_reduce[i]
    point_0 = np.array([x0, y0])
    cost = math.sqrt((point_0[0] - point_1[0])**2 + (point_0[1] - point_1[1])**2)
    Cost.append(cost)
Cost = np.array(Cost)

##循环发送当前扫掠的位置坐标，接收扫掠图像，根据角点拓扑确定终止点
cover_copy = cover_reduce
Cost_copy = Cost
boundary_all = []
count_usable = []
while True:
    index = np.argmin(Cost_copy)
    [x_current, y_current] = cover_copy[index]
    # 向机械臂发送此时的位置信号
    location = [x_current + h/2, y_current - w/2]

    # 移除已经扫掠过的区域
    index_origin = int(np.where((Cost==Cost_copy[index]))[0]) # 检索当前扫掠的区域的编号，为接下来的配准做准备
    count_usable.append(index_origin)
    Cost_copy = np.delete(Cost_copy, index)
    cover_copy.remove([x_current, y_current])
    # 根据返回的图像判断是否到达角点，到达角点的话删除角点外面的扫掠区域
    # # 去除噪点
    # boundary_unusable = np.loadtxt(r'object_pressure_meaurment/contour_unusable.txt')
    # # 读取轮廓点信息
    #
    # boundary_pixel_n = boundary_pixel_n.reshape(-1, 2)
    # boundary_unusable = set(map(tuple, boundary_unusable))
    # # 从A中去除出现在B中的坐标
    # boundary_pixel = [point for point in boundary_pixel_n if tuple(point) not in boundary_unusable]
    # # 将结果转换回NumPy数组格式
    # boundary_pixel = np.array(boundary_pixel)
    # if len(boundary_pixel) <= 1:
    #     continue
    # # 进行尺寸变换（需校准）
    # boundary_pixel_x = boundary_pixel[:, 0]
    # boundary_pixel_y = boundary_pixel[:, 1]
    # boundary_real_y = (boundary_pixel_x - 160) / 320 * dL + location[1]  # 注意此时像素的x对应实际的y坐标
    # boundary_real_x = (boundary_pixel_y - 120) / 240 * dH + location[0]
    # boundary_real = np.zeros_like(boundary_pixel)  # 机械臂坐标系下的边缘点坐标
    # boundary_real[:, 0] = boundary_real_x
    # boundary_real[:, 1] = boundary_real_y
    # # np.savetxt(r'object_pressure_meaurment/boundary_points_' + str(k) + '.txt', boundary_real)
    #
    # boundary_all.append(boundary_real)
    if len(Cost_copy) == 0: # 全部扫掠完毕，退出循环
        break

##根据重合区域进行配准，自动补全小范围缺失区域
for i in range(len(count_usable)):  # 进行误差的矫正
    flag = 0
    for j in range(len(count_usable)):
        [xt, yt, Wt, Ht, flag] = cover_overlap[count_usable[i], count_usable[j]]
        if flag == 0:
            continue
        boundary_real_1 = boundary_all[i]
        boundary_real_2 = boundary_all[j]
        if len(boundary_real_1) == 1 or len(boundary_real_2) == 1:
            continue
        boundary_filtered_1 = boundary_real_1[
            (boundary_real_1[:, 0] >= xt) & (boundary_real_1[:, 0] <= xt + Wt) & (boundary_real_1[:, 1] >= yt - Ht) & (
                    boundary_real_1[:, 1] <= yt)]  # 提取出第一个轮廓中符合条件的轮廓点集合
        boundary_filtered_2 = boundary_real_2[
            (boundary_real_2[:, 0] >= xt) & (boundary_real_2[:, 0] <= xt + Wt) & (boundary_real_2[:, 1] >= yt - Ht) & (
                    boundary_real_2[:, 1] <= yt)]  # 提取出第二个轮廓中符合条件的轮廓点集合
        if len(boundary_filtered_1) == 0 or len(boundary_filtered_2) == 0:
            continue  # 重合区域没有轮廓点，直接跳过
        boundary_filtered_1_mean = np.mean(boundary_filtered_1, axis=0)
        boundary_filtered_2_mean = np.mean(boundary_filtered_2, axis=0)
        # 根据重合区域质心的偏差，使用取半折中的方法对齐(这种方法对于图像不全的情况来说误差较大，需寻找新的办法)（更改成重合区域的轮廓重合程度最大）
        boundary_real_1_move = np.mean(
            np.concatenate((boundary_filtered_1_mean.reshape(-1, 2), boundary_filtered_2_mean.reshape(-1, 2)), axis=0),
            axis=0) - boundary_filtered_1_mean
        boundary_real_2_move = np.mean(
            np.concatenate((boundary_filtered_1_mean.reshape(-1, 2), boundary_filtered_2_mean.reshape(-1, 2)), axis=0),
            axis=0) - boundary_filtered_2_mean
        if Wt > Ht:  # 认为此时两个相邻区域是纵向相邻，只矫正横坐标
            boundary_real_1_move = np.array([boundary_real_1_move[0], 0])
            boundary_real_2_move = np.array([boundary_real_2_move[0], 0])
        if Wt < Ht:  # 认为此时两个相邻区域是横向相邻， 只矫正纵坐标
            boundary_real_1_move = np.array([0, boundary_real_1_move[1]])
            boundary_real_2_move = np.array([0, boundary_real_2_move[1]])
        if np.any(abs(boundary_real_1_move) > 0.99) or np.any(abs(boundary_real_2_move) > 0.99):
            print("[%f]和[%f]误差过大" % (i, j))
            continue

        boundary_real_1 = boundary_real_1 + boundary_real_1_move
        boundary_real_2 = boundary_real_2 + boundary_real_2_move
        boundary_all[i] = boundary_real_1
        boundary_all[j] = boundary_real_2  # 消除偏差的方法还有待进一步改进，主要是会让一块接触区域多次平移

##补偿缺失区域，并进行最终配准
