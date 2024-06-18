import cv2
import numpy as np
from contact_detection import calibration
import math
from image_flow import Flow
from image_boundary import boundary_depth, choose_boundary


def depth_boundary_detection(img):
    img_small = cv2.resize(img, (int(img.shape[1]), int(img.shape[0])))
    dm = nn.get_depthmap(img_small, mask_markers=True)
    # 保存关键点
    contour_points = []

    # 高斯滤波
    ksize = 5  # 高斯核大小
    sigma = 2.2  # 高斯核标准差，越小保留的细节越多
    gray_blur = cv2.GaussianBlur(dm, (ksize, ksize), sigma)
    # 计算x方向的梯度
    gradient_x = cv2.Sobel(dm, cv2.CV_64F, 1, 0, ksize=3)
    # 计算y方向的梯度
    gradient_y = cv2.Sobel(dm, cv2.CV_64F, 0, 1, ksize=3)
    # 计算梯度幅值和方向
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)
    # 阈值化边缘图像
    threshold = 2
    edges_dm = cv2.threshold(gradient_magnitude, threshold, 255, cv2.THRESH_BINARY)[1]
    # 将图像转换为8位单通道的二值图像
    edges_dm = cv2.convertScaleAbs(edges_dm)
    # 轮廓提取
    contours, _ = cv2.findContours(edges_dm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0 and nn.flag == 1:
        # 从最大的轮廓开始保留，直到轮廓点的保留数达到总轮廓点的90%以上
        total_points = 0
        preserved_contours = []
        for contour in sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True):
            if total_points / len(np.concatenate(contours)) < 0.9:
                preserved_contours.append(contour)
                total_points += len(contour)
            else:
                break
        cv2.drawContours(img_small, np.concatenate(preserved_contours), -1, (0, 0, 255), 2)
        print('len_contours=', len(preserved_contours))
        # 保存轮廓点到指定路径
        for contour in preserved_contours:
            for point in contour:
                contour_points.append(point[0])
        contour_points_array = np.array(contour_points)
        # 消除保存的数组内所有相同坐标点，只保留所有不同的坐标点
        contour_points_array = np.unique(contour_points_array, axis=0)
        # np.savetxt(r'first time/contour_points_8.txt', contour_points_array)
    cv2.imshow('Edges', edges_dm)
    cv2.imshow('Gradient Magnitude', gradient_magnitude)
    cv2.imshow('contours_img', img_small)

    return contour_points

def img_boundary_detection(img):
    # 将图像转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    # 高斯滤波
    ksize = 5  # 高斯核大小
    sigma = 1.2  # 高斯核标准差，越小保留的细节越多
    gray_blur = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    # ref_gray_blur = cv2.GaussianBlur(ref_gray, (ksize, ksize), sigma)

    # 计算差异图像
    diff = gray_blur

    # 基于梯度的边缘检测
    gradient_x = cv2.Sobel(diff, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(diff, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)
    gradient_magnitude = np.uint8(gradient_magnitude)

    # 阈值化边缘图像
    threshold = 20
    edges = cv2.threshold(gradient_magnitude, threshold, 255, cv2.THRESH_BINARY)[1]

    # 进行形态学操作来减小噪声
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 轮廓提取
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤小面积轮廓
    min_contour_area = 10  # 最小轮廓面积阈值
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]

    if len(filtered_contours) > 0:
        # 绘制过滤后的轮廓，显示在全黑的背景上
        mask = np.zeros_like(edges)
        cv2.drawContours(mask, filtered_contours, -1, 255, thickness=cv2.FILLED)

        # 将原边缘图像与绘制轮廓后的背景图像做“与”运算，得到在轮廓内的边缘点
        edges_in_filtered = cv2.bitwise_and(edges, mask)

        # 合并轮廓
        merged_contour = np.concatenate(filtered_contours)

        # 计算凸包
        hull = cv2.convexHull(merged_contour)

        # 绘制轮廓
        contour_img = np.zeros_like(img)
        # cv2.drawContours(img, [hull], -1, (0, 0, 255), 2)

        # Combine the original image with the filtered edges
        # result_img = cv2.addWeighted(img, 0.7, cv2.cvtColor(edges_in_filtered, cv2.COLOR_GRAY2BGR), 0.3, 0)

        # 显示结果
        cv2.imshow('Original Image', img)
        cv2.imshow('Edges', edges_in_filtered)

    # 显示结果
    else:
        cv2.imshow('Original Image', img)
        cv2.imshow('Edges', edges)

    return filtered_contours

# 读取GelSight图像
cap = cv2.VideoCapture(
    r'first time/wujiaoxing.avi')  # choose to read from video or camera
# cap = cv2.VideoCapture(0)
# image = cv2.imread(r'USB_bigforce_horizoninsertion/flow_init picture/flow_init178.png', cv2.IMREAD_COLOR)
reference = cv2.imread(r'first time/flow_init4.png', cv2.IMREAD_COLOR)
ret, img = cap.read()

nn = boundary_depth(cam_id=0)

Flow = Flow(col=320, row=240)

# # 提取接触表面的轮廓
# # 还需提取出关键特征点，并计算关键特征点的实际位置

while ret and cv2.waitKey(1) == -1:
    img = Flow.get_raw_img(img)
    #使用原始图像加滤波的方式提取边缘轮廓
    # contour_points_img = img_boundary_detection(img)

    # 使用深度信息对边缘进行过滤
    contour_points_depth = depth_boundary_detection(img)

    ret, img = cap.read()
# cv2.waitKey(0)
cv2.destroyAllWindows()
