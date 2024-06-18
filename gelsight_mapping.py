import cv2
import numpy as np
from contact_detection import calibration
import math
from image_flow import Flow
from image_boundary import boundary_depth, choose_boundary
import matplotlib.pyplot as plt


## 收集空间点坐标，进行缩放
def generate_pentagon(side_length):
    angles = np.deg2rad(np.arange(0, 360, 72))
    x = side_length * np.cos(angles)
    y = side_length * np.sin(angles)
    return np.column_stack((x, y))


def generate_star(pentagon, scale_factor):
    star_points = []
    for i in range(5):
        start_point = pentagon[i]
        end_point = pentagon[(i + 2) % 5]
        star_points.append(start_point)
        star_points.append(end_point)
    return np.array(star_points)


side_length = 25
scale_factor = 1

pentagon = generate_pentagon(side_length)
star_points = generate_star(pentagon, scale_factor)
# 逆时针方向保存10个顶点的坐标
points_location = star_points[::2]

# points_location = [[531.0, -34.28], [541.85, -48.95], [542.96, -34.35], [542.36, -19.56], [548.27, -41.58], [554.49, -34.3],
#             [549.34, -26.97], [553.99, -41.59], [553.87, -26.91], [561.35, -41.76], [560.35, -26.93]] #深度相机收集到的在机械臂坐标系下的轮廓点坐标
points_location_mean = np.mean(points_location, axis=0)  # 求解区域的质心
scale = 1.1  # 缩放比例
outer_points = points_location_mean + scale * (points_location - points_location_mean)  # 得到最终的缩放后的轮廓点坐标
scale = 0.9  # 缩放比例
inner_points = points_location_mean + scale * (points_location - points_location_mean)  # 得到最终的缩放后的轮廓点坐标

## 使用最小矩形框框住指定区域
# x, y, W, H = cv2.boundingRect(np.array(outer_points, dtype=np.float32))
x = 529
y = -229
W = 73
H = 74
# bottom_left = (x, y) #矩形框左上下角坐标
# top_right = (x + W, y + H) #矩形框左右上角坐标

## 划分初始接触区域
dh = 12  # 纵向接触有效区域
dw = 14  # 横向接触有效区域
n = H // dh + np.sign(H % dh)  # 纵向覆盖矩形框所需有效接触区域个数
m = W // dw + np.sign(W % dw)  # 横向覆盖矩形框所需有效接触区域个数
if H > 2 * dh:
    dy = (H - (n - 2) * dh) / 2
else:
    dy = 0
if W > 2 * dw:
    dx = (W - (m - 2) * dw) / 2
else:
    dx = 0
cover = np.zeros((int(n), int(m), 2))  # 初始覆盖满最小矩形框所需的接触区域数目为：(n)*(m)个
cover[0, 0] = np.array([x, y + H])  # 左上角矩形为起点(选取的特征点为接触区域矩形的左上点)
cover[0, m - 1] = np.array([x + W - dw, y + H])
cover[n - 1, 0] = np.array([x, y + dh])
cover[n - 1, m - 1] = np.array([x + W - dw, y + dh])  # 右下矩形为终点
for i in range(1, n - 1):
    cover[i, 0] = np.array([x, y + H - dy - (i - 1) * dh])
    cover[i, m - 1] = np.array([x + W - dw, y + H - dy - (i - 1) * dh])
    for j in range(1, m - 1):
        cover[0, j] = np.array([x + dx + (j - 1) * dw, y + H])
        cover[n - 1, j] = np.array([x + dx + (j - 1) * dw, y + dh])
        cover[i, j] = np.array([x + dx + (j - 1) * dw, y + H - dy - (i - 1) * dh])  # cover保存的矩形坐标点为其左上角点

## 调整接触区域
cover_reduce = []
for i in range(n):
    for j in range(m):
        x = cover[i, j, 0]
        y = cover[i, j, 1]
        # # 检查矩形是否与环形区域有重叠部分
        # # 将轮廓坐标转换为OpenCV需要的格式
        # inner_contour = np.array(inner_points, dtype=np.int32)
        # outer_contour = np.array(outer_points, dtype=np.int32)
        # # 将矩形表示为一个轮廓
        # rect_contour = np.array([(x, y), (x, y - dh), (x + dw, y - dh), (x + dw, y)],
        #                         dtype=np.int32)  # 因为特征点选取的为接触区域的左上点，故可推出矩形的四个角点坐标
        # # 检测矩形与环形区域的轮廓是否相交(还有待进一步完善)
        # retval_inner, intersection = cv2.intersectConvexConvex(rect_contour, inner_contour)
        # retval_outer, intersection = cv2.intersectConvexConvex(rect_contour, outer_contour)
        # if retval_inner > 0 or retval_outer > 0:
        #     cover_reduce.append([x, y])  # 把与环形区域相交的接触区域矩形提取出来
        cover_reduce.append([x, y])

# # 根据重合区域的图像减少误差(这里把接触的位置坐标发给机械臂，然后再把机械臂的实际位置返回作为cover_reduce)
# # 实际的机械臂追踪位置
# wujiaoxing_pressure_meaurment_3的坐标数据
# cover_center = [[530.64, -22.01], [538.83, -22.0], [554.61, -21.95], [565.39, -22.03], [529.84, -28.79],
#                 [540.23, -28.91],
#                 [555.79, -28.87], [564.9, -28.91], [530.00, -40.64], [538.79, -40.64], [555.28, -40.96],
#                 [564.53, -40.71],
#                 [529.88, -52.34], [538.95, -52.36], [555.32, -52.4], [564.84, -52.45], [545.10, -52.32]]
# # wujiaoxing_pressure_meaurment_2的坐标数据
# # cover_center = [[531.0, -34.28], [541.85, -48.95], [542.96, -34.35], [542.36, -19.56], [548.27, -41.58],
# #                 [554.49, -34.3],
# #                 [549.34, -26.97], [553.99, -41.59], [553.87, -26.91], [561.35, -41.76], [560.35, -26.93]]
# # wubianxing
# cover_center = [[536.73, -140.78], [545.97, -140.33], [532.31, -154.78], [546.45, -165.97], [555.91, -165.86],
#                 [566.8, -153.24], [556.36, -140`.43], [565.86, -142.28], [565.72, -161.98]]
# # liubianxing
# cover_center = [[532.42, -201.05], [546.70, -201.25], [561.68, -201.46], [532.96, -212.93], [560.70, -213.10],
#                 [540.11, -221.70], [547.68, -220.31], [560.76, -222.11], [541.76, -192.21], [558.01, -195.65], [569.65, -208.81]]
# # object
# cover_center = [[576.09, -78.54], [568.94, -78.53], [568.89, -53.94], [578.98, -53.99], [589.57, -54.09]]

# cover_reduce = []
# for k in range(len(cover_center)):
#     x = cover_center[k][0] - 6
#     y = cover_center[k][1] + 7  # 进行逆变换时一定注意，增减量
#     cover_reduce.append([x, y])
#
# cover_overlap = np.zeros((len(cover_reduce), len(cover_reduce), 5))  # 定义接触区域之间两两重合的区域，每个区域表示为（x,y,w,h, flag）五维形式
# for i in range(len(cover_reduce) - 1):
#     [x1, y1] = cover_reduce[i]
#     for j in range(i + 1, len(cover_reduce)):
#         [x2, y2] = cover_reduce[j]
#         if max(x1, x2) - min(x1, x2) > dw or max(y1, y2) - min(y1, y2) > dh:
#             cover_overlap[i, j] = np.array([0, 0, 0, 0, 0])
#             continue
#         xt = max(x1, x2)
#         yt = min(y1, y2)
#         Wt = min(x1, x2) + dw - xt
#         Ht = yt + dh - max(y1, y2)
#         cover_overlap[i, j] = np.array([xt, yt, Wt, Ht, 1])  # 这里要再把不相邻区域之间的接触给去掉

cover_center = cover_reduce  # 机械臂需运动到的中心点
for i in range(len(cover_reduce)):
    x = cover_reduce[i][0] + 6
    y = cover_reduce[i][1] - 7
    cover_center[i] = [x, y]

## 画出在机械臂坐标系中的轮廓图像
location = cover_center
boundary_all = []
count_usable = []

# 去除噪点
boundary_unusable = np.loadtxt(r'wujiaoxing_pressure_meaurment/contour_unusable.txt')
for k in range(len(cover_reduce)):
    # 读取轮廓点信息
    file_name = r'wujiaoxing_pressure_meaurment/contour_points_' + str(k + 1) + '.txt'
    boundary_pixel_n = np.loadtxt(file_name)  # 下载边缘像素点的坐标
    boundary_pixel_n = boundary_pixel_n.reshape(-1, 2)
    boundary_unusable = set(map(tuple, boundary_unusable))
    # 从A中去除出现在B中的坐标
    boundary_pixel = [point for point in boundary_pixel_n if tuple(point) not in boundary_unusable]
    # 将结果转换回NumPy数组格式
    boundary_pixel = np.array(boundary_pixel)
    if len(boundary_pixel) <= 1:
        continue
    # 进行尺寸变换（需校准）
    dL = 20.42  # 横坐标（宽度）对应的尺度变换 （mm)
    dH = 15.0  # 纵坐标（高度）对应的尺度变换（mm）
    boundary_pixel_x = boundary_pixel[:, 0]
    boundary_pixel_y = boundary_pixel[:, 1]
    boundary_real_y = (boundary_pixel_x - 160) / 320 * dL + location[k][1]  # 注意此时像素的x对应实际的y坐标
    boundary_real_x = (boundary_pixel_y - 120) / 240 * dH + location[k][0]
    boundary_real = np.zeros_like(boundary_pixel)  # 机械臂坐标系下的边缘点坐标
    boundary_real[:, 0] = boundary_real_x
    boundary_real[:, 1] = boundary_real_y
    # np.savetxt(r'object_pressure_meaurment/boundary_points_' + str(k) + '.txt', boundary_real)

    boundary_all.append(boundary_real)
    count_usable.append(k)

# count_already = []
# for i in range(len(count_usable)):  # 进行误差的矫正
#     flag = 0
#     for j in range(len(count_usable)):
#         [xt, yt, Wt, Ht, flag] = cover_overlap[count_usable[i], count_usable[j]]
#         if flag == 0:
#             continue
#         boundary_real_1 = boundary_all[i]
#         boundary_real_2 = boundary_all[j]
#         if len(boundary_real_1) == 1 or len(boundary_real_2) == 1:
#             continue
#         boundary_filtered_1 = boundary_real_1[
#             (boundary_real_1[:, 0] >= xt) & (boundary_real_1[:, 0] <= xt + Wt) & (boundary_real_1[:, 1] >= yt - Ht) & (
#                     boundary_real_1[:, 1] <= yt)]  # 提取出第一个轮廓中符合条件的轮廓点集合
#         boundary_filtered_2 = boundary_real_2[
#             (boundary_real_2[:, 0] >= xt) & (boundary_real_2[:, 0] <= xt + Wt) & (boundary_real_2[:, 1] >= yt - Ht) & (
#                     boundary_real_2[:, 1] <= yt)]  # 提取出第二个轮廓中符合条件的轮廓点集合
#         if len(boundary_filtered_1) == 0 or len(boundary_filtered_2) == 0:
#             continue  # 重合区域没有轮廓点，直接跳过
#         boundary_filtered_1_mean = np.mean(boundary_filtered_1, axis=0)
#         boundary_filtered_2_mean = np.mean(boundary_filtered_2, axis=0)
#         # 根据重合区域质心的偏差，使用取半折中的方法对齐(这种方法对于图像不全的情况来说误差较大，需寻找新的办法)
#         boundary_real_1_move = np.mean(
#             np.concatenate((boundary_filtered_1_mean.reshape(-1, 2), boundary_filtered_2_mean.reshape(-1, 2)), axis=0),
#             axis=0) - boundary_filtered_1_mean
#         boundary_real_2_move = np.mean(
#             np.concatenate((boundary_filtered_1_mean.reshape(-1, 2), boundary_filtered_2_mean.reshape(-1, 2)), axis=0),
#             axis=0) - boundary_filtered_2_mean
#         if Wt > Ht:  # 认为此时两个相邻区域是纵向相邻，只矫正横坐标
#             boundary_real_1_move = np.array([boundary_real_1_move[0], 0])
#             boundary_real_2_move = np.array([boundary_real_2_move[0], 0])
#         if Wt < Ht:  # 认为此时两个相邻区域是横向相邻， 只矫正纵坐标
#             boundary_real_1_move = np.array([0, boundary_real_1_move[1]])
#             boundary_real_2_move = np.array([0, boundary_real_2_move[1]])
#         if np.any(abs(boundary_real_1_move) > 0.99) or np.any(abs(boundary_real_2_move) > 0.99):
#             print("[%f]和[%f]误差过大" % (i, j))
#             continue
#         ## 新方法
#
#
#         boundary_real_1 = boundary_real_1 + boundary_real_1_move
#         boundary_real_2 = boundary_real_2 + boundary_real_2_move
#         boundary_all[i] = boundary_real_1
#         boundary_all[j] = boundary_real_2  # 消除偏差的方法还有待进一步改进，主要是会让一块接触区域多次平移
#         count_already.append(i)
#         count_already.append(j)

# boundary_all_new = []
# for k in range(len(boundary_all)):
#     boundary = boundary_all[k]
#     indices = np.where((boundary[:,1] <= -82))
#     boundary = np.delete(boundary, indices, axis=0)
#     indices = np.where((boundary[:, 0] <= 582 - 12.1))
#     boundary = np.delete(boundary, indices, axis=0)
#     boundary_all_new.append(boundary)
# boundary_all = boundary_all_new

for k in range(len(boundary_all)):
    plt.scatter(boundary_all[k][:, 0], boundary_all[k][:, 1], alpha=0.7)

plt.legend()
# plt.xlim(539.1, 579.1)
# plt.ylim(-57.5, -17.5)
plt.xticks(fontsize=18, fontproperties='Times New Roman')
plt.yticks(fontsize=18, fontproperties='Times New Roman')
plt.xlabel('X(mm)', fontsize=20, fontdict={'family': 'Times New Roman'})
plt.ylabel('Y(mm)', fontsize=20, fontdict={'family': 'Times New Roman'})
plt.title('Double axis bolt', fontsize=20, fontdict={'family': 'Times New Roman'})
plt.show()


def is_point_adjacent(star_boundary, x, y):
    k=0
    for point in star_boundary:
        if ((point[0] - x)**2 + (point[1] - y)**2 == 1) and (point[0] != x or point[1] != y):
            k += 1
    return k

def is_point_in_rectangle(point, x, y, x1, y1):
    px, py = point
    x_min = min(x, x1)
    x_max = max(x, x1)
    y_min = min(y, y1)
    y_max = max(y, y1)

    if x_min == x_max:
        return y_min < py < y_max
    elif y_min == y_max:
        return x_min < px < x_max
    else:
        return x_min < px < x_max and y_min < py < y_max
# boundary_flattened = [item for sublist in boundary_all for item in sublist]
# star_boundary = np.array(boundary_flattened).reshape(-1, 2)
# star_boundary = star_boundary.astype(np.int32)
# star_boundary = np.unique(star_boundary, axis=0)
star_boundary = np.loadtxt(r'wujiaoxing_pressure_meaurment/star_hole_boundary.txt')
# star_boundary[:, 0] = star_boundary[:, 0]
## 补全缺失的中断点
#判断中断点
abrupt_point = []
for i in range(len(star_boundary)):
    x = star_boundary[i,0]
    y = star_boundary[i,1]
    n_near = is_point_adjacent(star_boundary, x, y)
    if n_near < 2:
        abrupt_point.append([x,y])
#确定中断线段的始终点
line_start_end = []
for i in range(len(abrupt_point)-1):
    [x, y] = abrupt_point[i]
    for j in range(i+1, len(abrupt_point)):
        [x1, y1] = abrupt_point[j]
        points_in_rectangle = [point for point in star_boundary if is_point_in_rectangle(point, x, y, x1, y1)]
        if len(points_in_rectangle) == 0:
            line_start_end.append([x, y, x1, y1])
# 连接中断点
line_add = []
for i in range(len(line_start_end)):
    [x, y, x1, y1] = line_start_end[i]
    if abs(x1 - x)>abs(y1 - y):
        direction = 0
    else:
        direction = 1
    while True:
        delta_x = np.sign(x1 - x)
        delta_y = np.sign(y1 - y)
        distance_x = np.sqrt((x1 - x - delta_x)**2 + (y1 - y)**2)
        distance_y = np.sqrt((x1 - x)**2 + (y1 - y - delta_y)**2)
        if distance_x < distance_y:
            x = x + delta_x
            direction = 1
        elif distance_y < distance_x:
            y = y + delta_y
            direction = 0
        else:
            if direction == 0:
                x = x + delta_x
                direction = 1
            else:
                y = y + delta_y
                direction = 0
        line_add.append([x, y])
        if np.sqrt((x1 - x)**2 + (y1 - y)**2) == 1:
            break
# line_add = np.array(line_add).reshape(-1, 2)
# star_boundary = np.append(star_boundary, line_add).reshape(-1,2)

from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from scipy.interpolate import make_interp_spline
from scipy.interpolate import CubicSpline

## 拟合多边形
# 进行GMM聚类
n_clusters = 10  # 聚类数目
gmm = GaussianMixture(n_components=n_clusters)
gmm.fit(star_boundary)
# 预测每个样本所属的聚类
cluster_labels = gmm.predict(star_boundary)
cluster_detail = gmm.predict_proba(star_boundary)
# 获取每个聚类的均值、协方差和权重
cluster_means = gmm.means_
cluster_covs = gmm.covariances_
cluster_weights = gmm.weights_
fig, ax = plt.subplots(1, 1)
# plt.figure()
for k in range(n_clusters):
    ax.scatter(star_boundary[cluster_labels == k, 0], star_boundary[cluster_labels == k, 1], label=f'Cluster {k+1}', alpha=0.7)
    cov_matrix = cluster_covs[k][[0, 1]][:, [0, 1]]
    v, w = np.linalg.eigh(cov_matrix)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # Convert eigenvalues to radii
    angle = np.arctan(w[0, 1] / w[0, 0])
    ell = Ellipse(cluster_means[k, [0, 1]], v[0], v[1], 180.0 * angle / np.pi, color='black', alpha=0.2)
    # ax.add_patch(ell)

# plt.xlim(529.1, 579.1)
# plt.ylim(-60.5, -16.5)
plt.xticks(fontsize=18, fontproperties='Times New Roman')
plt.yticks(fontsize=18, fontproperties='Times New Roman')
plt.xlabel('X(mm)', fontsize=20, fontdict={'family': 'Times New Roman'})
plt.ylabel('Y(mm)', fontsize=20, fontdict={'family': 'Times New Roman'})
# plt.title('Double axis bolt', fontsize=20, fontdict={'family': 'Times New Roman'})
# plt.legend()

# 拟合每一类为一直线
param = []


# 定义直线函数模型（y = mx + b）
def linear_model(x, m, b):
    return m * x + b


def intersection_point(m1, b1, m2, b2):
    if m1 == m2:
        return None
    else:
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1
        return x, y


# # 为每个类别拟合直线并绘制线段
for k in range(n_clusters):
    x = star_boundary[cluster_labels == k, 0]
    y = star_boundary[cluster_labels == k, 1]

    # 使用线性回归拟合直线
    param,covariance = curve_fit(linear_model, x, y)

    X = np.arange(min(x), max(x), 0.05)
    Y = linear_model(X, *param)
    # plt.plot(X, Y)
#
#     # 找到与其他直线相交的点
#     for k1 in range(n_clusters):
#         if k1 != k:
#             other_x = star_boundary[cluster_labels == k1, 0]
#             other_y = star_boundary[cluster_labels == k1, 1]
#             other_m, other_b = curve_fit(linear_model, other_x, other_y)
#
#             # 计算交点
#             intersection_x = (other_b - b) / (m - other_m)
#             intersection_y = m * intersection_x + b
#
#             # 截断线段并绘制
#             x = np.append(x, intersection_x)
#             y = np.append(y, intersection_y)
#
#     # 对 x 值进行排序以绘制有序线段
#     sorted_indices = np.argsort(x)
#     plt.plot(x[sorted_indices], y[sorted_indices], label=f'Cluster {k}')
#
# plt.legend()
# plt.grid()
plt.show()
# plt.figure()