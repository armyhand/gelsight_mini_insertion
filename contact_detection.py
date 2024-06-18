import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from mpl_toolkits.mplot3d import Axes3D
# from scipy import signal



class calibration:
    def __init__(self):
        self.BallRad = 7.6 / 2  # 4.76/2 #mm
        self.Pixmm = .10577  # .10577 #4.76/100 #0.0806 * 1.5 mm/pixel
        self.ratio = 1 / 2.
        self.red_range = [-90, 90]
        self.green_range = [-90, 90]  # [-60, 50]
        self.blue_range = [-90, 90]  # [-80, 60]
        self.red_bin = int((self.red_range[1] - self.red_range[0]) * self.ratio)
        self.green_bin = int((self.green_range[1] - self.green_range[0]) * self.ratio)
        self.blue_bin = int((self.blue_range[1] - self.blue_range[0]) * self.ratio)
        self.zeropoint = [-90, -90, -90]
        self.lookscale = [180., 180., 180.]
        self.bin_num = 90
        self.abe_array = np.load('abe_corr.npz')  # change this with your aberration array
        self.x_index = self.abe_array['x']
        self.y_index = self.abe_array['y']

    def get_raw_img(self, frame, img_width, img_height):
        img = cv2.resize(frame, (895, 672))  # size suggested by janos to maintain aspect ratio
        border_size_x, border_size_y = int(img.shape[0] * (1 / 7)), int(
            np.floor(img.shape[1] * (1 / 7)))  # remove 1/7th of border from each size
        img = img[border_size_x:img.shape[0] - border_size_x, border_size_y:img.shape[1] - border_size_y]
        img = img[:, :-1]  # remove last column to get a popular image resolution
        img = cv2.resize(img, (img_width, img_height))  # final resize for 3d

        return img

    def crop_image(self, img, pad):
        return img[pad:-pad, pad:-pad]

    def mask_marker(self, raw_image):
        m, n = raw_image.shape[1], raw_image.shape[0]
        raw_image = cv2.pyrDown(raw_image).astype(np.float32)
        blur = cv2.GaussianBlur(raw_image, (25, 25), 0)
        blur2 = cv2.GaussianBlur(raw_image, (5, 5), 0)
        diff = blur - blur2
        diff *= 16.0
        # cv2.imshow('blur2', blur.astype(np.uint8))
        # cv2.waitKey(1)

        diff[diff < 0.] = 0.
        diff[diff > 255.] = 255.

        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        # cv2.imshow('diff', diff.astype(np.uint8))
        # cv2.waitKey(1)

        mask_b = diff[:, :, 0] > 150
        mask_g = diff[:, :, 1] > 150
        mask_r = diff[:, :, 2] > 150
        mask = (mask_b * mask_g + mask_b * mask_r + mask_g * mask_r) > 0
        # cv2.imshow('mask', mask.astype(np.uint8) * 255)
        # cv2.waitKey(1)
        mask = cv2.resize(mask.astype(np.uint8), (m, n))
        #        mask = mask * self.dmask
        #        mask = cv2.dilate(mask, self.kernal4, iterations=1)

        # mask = cv2.erode(mask, self.kernal4, iterations=1)
        return (1 - mask) * 255

    def find_dots(self, binary_image):
        # down_image = cv2.resize(binary_image, None, fx=2, fy=2)
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 1
        params.maxThreshold = 12
        params.minDistBetweenBlobs = 9
        params.filterByArea = True
        params.minArea = 9
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.minInertiaRatio = 0.5
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(binary_image.astype(np.uint8))
        # im_to_show = (np.stack((binary_image,)*3, axis=-1)-100)
        # for i in range(len(keypoints)):
        #     cv2.circle(im_to_show, (int(keypoints[i].pt[0]), int(keypoints[i].pt[1])), 5, (0, 100, 100), -1)
        #
        # cv2.imshow('final_image1',im_to_show)
        # cv2.waitKey(1)
        return keypoints

    def make_mask(self, img, keypoints):
        img = np.zeros_like(img[:, :, 0])
        for i in range(len(keypoints)):
            # cv2.circle(img, (int(keypoints[i].pt[0]), int(keypoints[i].pt[1])), 6, (1), -1)
            cv2.ellipse(img, (int(keypoints[i].pt[0]), int(keypoints[i].pt[1])), (9, 7), 0, 0, 360, (1), -1)

        # cv2.imshow('final_image2', img)
        # cv2.waitKey(1)
        return img

    def contact_detection(self, raw_image, ref, marker_mask):
        blur = cv2.GaussianBlur(ref.astype(np.float32), (3, 3), 0)
        diff_img = np.max(np.abs(raw_image.astype(np.float32) - blur), axis=2)
        contact_mask = (diff_img > 30).astype(np.uint8) * (1 - marker_mask)
        contours, _ = cv2.findContours(contact_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 检测轮廓，返回轮廓的向量坐标值
        if len(contours) <= 5:
            # cv2.imshow("contact", raw_image)
            ellipse = None
        else:
            areas = [cv2.contourArea(c) for c in contours]
            areas = np.array(areas)
            index = np.where(areas >= 1500)
            index = np.array(index).reshape(-1)
            # sorted_areas = np.sort(areas)
            # selected_contours = contours[areas.index(sorted_areas[-1])]  # the biggest contour
            # selected_contours = np.append(contours[areas.index(sorted_areas[-1])], contours[areas.index(sorted_areas[-2])]).reshape(-1, 1, 2)
            selected_contours = [contours[i] for i in index]
            all_points = np.concatenate(selected_contours)

            cv2.drawContours(raw_image, selected_contours, -1, (0, 0, 255), 1) #画轮廓需用列表，列表中每一项为一个ndarray数组，表示一个轮廓
            # (x, y), radius = cv2.minEnclosingCircle(cnt)
            ellipse = cv2.fitEllipse(all_points) #拟合椭圆需用ndarray数组

            # center = (int(x), int(y))
            # radius = int(radius)
            # im2show = cv2.circle(np.array(raw_image), center, radius, (0, 40, 0), 2)
            # im2show = cv2.ellipse(raw_image, ellipse, (0, 255, 0), 2)

        # contact_mask = contact_mask * (1 - marker_mask)
        # contact_mask = np.zeros_like(contact_mask)
        # cv2.circle(contact_mask, center, radius, (1), -1)
        # cv2.imshow('contact_mask',contact_mask*255)
        # cv2.waitKey(0)
        return ellipse

if __name__ == "__main__":
    cali = calibration()
    pad = 20
    ref_img = cv2.imread('test_data_2/ref.jpg')
    ref_img = ref_img[cali.x_index, cali.y_index, :]
    ref_img = cali.crop_image(ref_img, pad)
    marker = cali.mask_marker(ref_img)
    keypoints = cali.find_dots(marker)
    marker_mask = cali.make_mask(ref_img, keypoints)
    marker_image = np.dstack((marker_mask, marker_mask, marker_mask))
    ref_img = cv2.inpaint(ref_img, marker_mask, 3, cv2.INPAINT_TELEA)
    # table = np.zeros((cali.blue_bin, cali.green_bin, cali.red_bin, 2))
    # table_account = np.zeros((cali.blue_bin, cali.green_bin, cali.red_bin))
    # cv2.imshow('ref_image', ref_img)
    # cv2.waitKey(0)
    has_marke = True

    cap = cap = cv2.VideoCapture(
    r'USB_bigforce_horizoninsertion/flow_init picture/flow_init.avi')  # choose to read from video or camera
    # cap = cv2.VideoCapture(0)
    success, img = cap.read()
    while success and cv2.waitKey(1) == -1:
        # img = cali.get_raw_img(img, 640, 480)
        img = img[cali.x_index, cali.y_index, :]
        img = cali.crop_image(img, pad)
        if has_marke:
            marker = cali.mask_marker(img)
            keypoints = cali.find_dots(marker)
            marker_mask = cali.make_mask(img, keypoints)
        else:
            marker_mask = np.zeros_like(img)
        contours = cali.contact_detection(img, ref_img, marker_mask)
        if contours == None:
            cv2.imshow("contact", img)
        else:
            im2show = cv2.ellipse(img, contours, (0, 255, 0), 2)
            cv2.imshow('contact', im2show)

        if cv2.waitKey(40) & 0xFF == ord('q'):
            break
        success, img = cap.read()

    cv2.destroyWindow("contact")
    cap.release()