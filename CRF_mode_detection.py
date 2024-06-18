#### 使用PCA方法进行传感图像分析
##自适应求K值
## 使用PCA方法提取完主成分后，采用GMM的方法进行无监督聚类（可以根据现有数据进行初步实践），再采用HMM进行监督分类（需要收集新的、带标签的接触状态数据）
import cv2
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from image_flow import Flow
import time
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
import matplotlib.font_manager as fm
import pickle
import socket
import multiprocessing
from client import send_mode_transfer

# 定义特征函数，用于将观察序列映射到特征
def word2features(sent, i):
    word = str(sent[i])
    number = str(i+1)
    prev_word = "<s>" if i == 0 else str(sent[i - 1])
    pprev_word = "<s>" if i <= 1 else str(sent[i - 2])
    next_word = "</s>" if i == (len(sent) - 1) else str(sent[i + 1])
    return {
        'w': word,
        'w-1': prev_word,
        'w-2': pprev_word,
        'w-1:w': prev_word + word,
        'w-2:w-1': pprev_word + prev_word,
        'w-2:w': pprev_word + word,
        'n': number,
        'bias': 1
    }

def word2labels(label, i, j):
    if j < 5:
        labels = 'y_axis:' + str(label[i])
    else:
        labels = 'x_axis:' + str(label[i])
    if label[i] == 1:
        labels = 'keepon:' + str(label[i])
    if label[i] == 2:
        labels = 'keepon:' + str(label[i])
    # if label[i] == 5:
    #     labels = 'keepon:' + str(label[i])
    if label[i] == 6:
        labels = 'keepon:' + str(label[i])
    return labels

if __name__=='__main__':
    # # 读取图像并转换为灰度图
    # images = [] #第一次运行时使用
    images = np.loadtxt(r'star_hole_y/images_allyx_3.txt')
    images = list(images)
    counts_all = []

    # # 读取光流视频进行分析
    # Flow = Flow(col=320, row=240)
    # for j in range(3): #使用循环把训练集的光流信息进行收集
    #     images = []  # 第一次运行时使用
    #     cap = cv2.VideoCapture(r'star_hole_x/human_demonstration_' + str(j+1) +'/flow_init picture/flow_init.avi') # choose to read from video or camera
    #     # 第二个传感器
    #     cap_2 = cv2.VideoCapture(r'star_hole_x/human_demonstration_' + str(j+1) +'/flow_init picture/flow_init_2.avi')
    #     time.sleep(1)
    #     # 第一个传感器
    #     ret, frame1 = cap.read()
    #     # frame1 = Flow.get_raw_img(frame1)
    #     prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    #     f0 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    #     # 第二个传感器
    #     ret, frame1_2 = cap_2.read()
    #     # frame1_2 = Flow.get_raw_img(frame1_2)
    #     prvs_2 = cv2.cvtColor(frame1_2, cv2.COLOR_BGR2GRAY)
    #     f0_2 = cv2.cvtColor(frame1_2, cv2.COLOR_BGR2GRAY)
    #
    #     hsv = np.zeros_like(frame1)
    #     hsv[..., 1] = 255
    #     flow_sum = np.zeros((Flow.row, Flow.col, 2))
    #     count = 0
    #     d = {}
    #     reset_threshold_error = 0.3  # ?
    #     reset_threshold_mean = 2  # ?
    #     while (1):
    #         # 第一个传感器
    #         try:
    #             ret, frame2 = cap.read()
    #             # frame2 = Flow.get_raw_img(frame2)
    #             next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    #             t1 = time.time()
    #         except:
    #             break
    #
    #         flow = cv2.calcOpticalFlowFarneback(f0, next, None, 0.5, 3, int(180 * Flow.col / 960), 5, 5, 1.2, 0)
    #
    #         flow_sum[:, :, 0] = flow[:, :, 0]
    #         flow_sum[:, :, 1] = flow[:, :, 1]
    #
    #         frame3 = np.copy(frame2)
    #         d_flow_1 = Flow.draw(frame3, flow_sum)  # frame3是此刻与初始时刻的光流图（校正后）
    #
    #         cv2.imshow('frame', frame3)
    #
    #         k = cv2.waitKey(30) & 0xff
    #         if k == 27:  # K=ESC
    #             break
    #         prvs = next
    #
    #         # 第二个传感器
    #         try:
    #             ret, frame2 = cap_2.read()
    #             # frame2 = Flow.get_raw_img(frame2)
    #             next_2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    #         except:
    #             break
    #
    #         flow = cv2.calcOpticalFlowFarneback(f0_2, next_2, None, 0.5, 3, int(180 * Flow.col / 960), 5, 5, 1.2, 0)
    #
    #         flow_sum[:, :, 0] = flow[:, :, 0]
    #         flow_sum[:, :, 1] = flow[:, :, 1]
    #
    #         frame3 = np.copy(frame2)
    #         d_flow_2 = Flow.draw(frame3, flow_sum)  # frame3是此刻与初始时刻的光流图（校正后）
    #
    #         cv2.imshow('frame2', frame3)
    #
    #         k = cv2.waitKey(30) & 0xff
    #         if k == 27:  # K=ESC
    #             break
    #         prvs_2 = next_2  #
    #
    #         d_flow = np.append(d_flow_1, d_flow_2)
    #         images.append(d_flow)
    #         t2 = time.time()
    #         print('freq=', 1.0 / (t2 - t1))
    #         count += 1
    #
    #     counts_all.append(count)
    #     cap.release()
    #     cap_2.release()
    #     cv2.destroyAllWindows()
    #
    #     np.savetxt(r'star_hole_x/huaman_demonsrtation_' + str(j+1) +'.txt', images) #保存光流信息
    # # 将图像块转换为一维向量
    image_vectors = [image.flatten() for image in images]

    # 数据标准化
    image_vectors = np.array(image_vectors)
    image_vectors = image_vectors.astype(np.float32)
    # image_new = np.zeros(0)
    # image_new = np.append(image_new, image_vectors[0:1204+252]).reshape(-1,560)
    # image_new = np.append(image_new, image_vectors[5365:]).reshape(-1,560)
    # mean = np.mean(image_new, axis=0)
    # std = np.std(image_new, axis=0)
    # image_new = (image_new - mean) / std

    mean = np.mean(image_vectors, axis=0)
    std = np.std(image_vectors, axis=0)
    image_vectors = (image_vectors - mean) / std
    # 进行快速PCA分析
    n_components = 5  # 选择要提取的主成分数量
    pca = PCA(n_components=n_components, svd_solver='randomized')
    pca.fit(image_vectors)
    # pca.fit(image_new)

    # 获取主成分和权重
    components = pca.components_  # 主成分特征值对应的特征向量
    weights = pca.transform(image_vectors)  # 降维后的数据(特征值)
    # weights_new = pca.transform(image_new)

    # 观察降维后的数据保留了原数据多少信息
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    # 可视化数据（画降维后的数据特征）
    for i in range(n_components):
        L = weights[:, i]
        t = np.arange(len(L))
        plt.subplot(int(n_components), 1, i + 1)
        plt.plot(t, L, lw=3.0, color='black')
        plt.ylabel(f'Weights {i+1}', fontsize=20, fontdict={'family': 'Times New Roman'})
        plt.xticks(fontsize=20, fontproperties='Times New Roman')
        plt.yticks(fontsize=20, fontproperties='Times New Roman')
    # plt.title('Principle Components Analysis', fontsize=20, fontdict={'family': 'Times New Roman'})
    plt.xlabel('Length', fontsize=22, fontdict={'family': 'Times New Roman'})
    plt.show()


    # 进行GMM聚类
    n_clusters = 8  # 聚类数目
    gmm = GaussianMixture(n_components=n_clusters)
    gmm.fit(weights)

    # 预测每个样本所属的聚类
    cluster_labels = gmm.predict(weights)
    cluster_detail = gmm.predict_proba(weights)
    # 获取每个聚类的均值、协方差和权重
    cluster_means = gmm.means_
    cluster_covs = gmm.covariances_
    cluster_weights = gmm.weights_

    # 两两组合绘制散点图和椭圆
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    combinations = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    times_font = fm.FontProperties(fname=r'Times_new_Roman.ttf', size=16)
    for idx, (i, j) in enumerate(combinations):
        ax = axs[idx // 3, idx % 3]
        plt.subplot(2, 3, idx+1)

        for k in range(n_clusters):
            plt.scatter(weights[cluster_labels == k, i], weights[cluster_labels == k, j], label=f'Cluster {k+1}', alpha=0.7)
            cov_matrix = cluster_covs[k][[i, j]][:, [i, j]]
            v, w = np.linalg.eigh(cov_matrix)
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # Convert eigenvalues to radii
            angle = np.arctan(w[0, 1] / w[0, 0])
            ell = Ellipse(cluster_means[k, [i, j]], v[0], v[1], 180.0 * angle / np.pi, color='black', alpha=0.2)
            ax.add_patch(ell)
            plt.xticks(fontsize=16, fontproperties='Times New Roman')
            plt.yticks(fontsize=16, fontproperties='Times New Roman')

        ax.set_xlabel(f'Principal Component {i + 1}', fontsize=18, fontdict={'family': 'Times New Roman'})
        ax.set_ylabel(f'Principal Component {j + 1}', fontsize=18, fontdict={'family': 'Times New Roman'})
        ax.set_title(f'PC{i + 1} vs PC{j + 1}', fontsize=18, fontdict={'family': 'Times New Roman'})
        ax.tick_params(axis='both', labelsize=16)
        ax.legend()

    plt.tight_layout()
    plt.show()
    #
    # 进行监督学习分类
    # 生成示例数据（观察序列和标签序列）
    X = cluster_labels
    Y = np.loadtxt(r'star_hole_y/labels_allyx_3.txt')

    # # 划分训练数据(star_hole_y + star_hole_x)
    # # X_new = [X[0:295], X[295:674], X[674:944], X[944:1204], X1[0:252], X1[252:475], X1[475:697], X1[697:928], X1[928:1153], X1[1444:1687], X1[1687:2015],
    # #          X1[2015:2237], X_reduce[0:213], X_reduce[213:432], X_reduce[432:655], X_reduce[655:956], X_reduce[1184:1399],
    # #          X_reduce[1652:1924], X_reduce[1924:], X_attr[0:236], X_attr[236:476], X_attr[476:746],
    # #          X_attr[746:]]
    # # Y_new = [Y[0:295], Y[295:674], Y[674:944], Y[944:1204], Y1[0:252], Y1[252:475], Y1[475:697], Y1[697:928], Y1[928:1153], Y1[1444:1687], Y1[1687:2015],
    # #          Y1[2015:2237], Y_reduce[0:213], Y_reduce[213:432], Y_reduce[432:655], Y_reduce[655:956], Y_reduce[1184:1399],
    # #          Y_reduce[1652:1924], Y_reduce[1924:], Y_attr[0:236], Y_attr[236:476], Y_attr[476:746],
    # #          Y_attr[746:]]
    #
    # # 划分训练数据（star_hole_y/images_allyx_3）
    X1 = X[1204:]
    Y1 = Y[1204:]
    X_reduce = X1[2237:4405]  # 划分训练数据(star_hole_x)
    Y_reduce = Y1[2237:4405]
    X_attr = X1[4405:]
    Y_attr = Y1[4405:]
    X_new = [X[0:295], X[295:674], X[674:944], X[944:1204], X1[0:252], X_reduce[1924:], X_attr[0:236], X_attr[236:476],
             X_attr[476:746],
             X_attr[746:]]
    Y_new = [Y[0:295], Y[295:674], Y[674:944], Y[944:1204], Y1[0:252], Y_reduce[1924:], Y_attr[0:236], Y_attr[236:476],
             Y_attr[476:746],
             Y_attr[746:]]
    #
    # # 划分训练数据（star_hole_x/images_allyx4）
    # count = [250, 211, 192, 379, 270, 295, 260, 257, 167, 140, 236, 240, 270, 277]
    # X_new = []
    # Y_new = []
    # for k in range(len(count)):
    #     X_add = X[sum(count[:(k)]): sum(count[:(k+1)])]
    #     Y_add = Y[sum(count[:(k)]): sum(count[:(k + 1)])]
    #     X_new.append(X_add)
    #     Y_new.append(Y_add)
    #
    # 将观察序列转换为特征序列
    X_features = [[word2features(x, i) for i in range(len(x))] for x in X_new]
    # Y_features = [[word2labels(y, i, k) for i in range(len(y))] for y in Y_new for k in range(len(Y_new))]
    k = 0
    Y_features = []
    for y in Y_new:
        Y_features.append([word2labels(y, i, k) for i in range(len(y))])
        k += 1

    # 创建CRF模型
    model = sklearn_crfsuite.CRF()

    # 训练模型
    model.fit(X_features, Y_features)

    # # 保存训练好的模型
    # model_filename = r'star_hole_y/crf_model_1.pkl'
    # with open(model_filename, 'wb') as model_file:
    #     pickle.dump(model, model_file)
    # # 加载训练好的模型
    # with open(model_filename, 'rb') as model_file:
    #     loaded_model = pickle.load(model_file)

    # 预测标签序列
    Y_pred = model.predict(X_features)

    # 打印预测结果和真实标签
    for i in range(len(X_features)):
        print("True:", Y_features[i])
        print("Predicted:", Y_pred[i])
        print()

    # 计算准确率
    accuracy = metrics.flat_accuracy_score(Y_features, Y_pred)
    print("Accuracy:", accuracy)
    for k in range(len(Y_features)):
        accuracy = metrics.flat_accuracy_score(Y_features[k], Y_pred[k])
        print('accuracy[%f]=%f' % (k + 1, accuracy))

    # # 重构纹理
    # # reconstructed_vectors = np.dot(weights[:, :n_components], components[:n_components, :])
    # # reconstructed_vectors = reconstructed_vectors + mean
    # # reconstructed_vectors = reconstructed_vectors.astype(np.uint8) # 转化成整数
    # # reconstructed_images = [reconstructed_vector.reshape(image.shape) for reconstructed_vector, image in zip(reconstructed_vectors, images)]
    # #
    # # # 显示重构后的图像
    # # for image in reconstructed_images:
    # #     cv2.imshow('Reconstructed Image', image)
    # #     cv2.waitKey(0)
    # # # cv2.imshow('Reconstructed Image', images[0])
    # #
    # # cv2.destroyAllWindows()

    ## 在线预测模式
    model_filename = r'star_hole_y/crf_model_3.pkl'
    # 加载训练好的模型
    with open(model_filename, 'rb') as model_file:
        crf_model = pickle.load(model_file)
    model_filename = r'star_hole_y/gmm_model_3.pkl'
    # 加载训练好的模型
    with open(model_filename, 'rb') as model_file:
        gmm_model = pickle.load(model_file)

    # 第一个传感器
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(r'star_hole_y/star_insertion_break_2/flow_init picture/flow_init.avi')

    # 第二个传感器
    cap_2 = cv2.VideoCapture(2)
    # cap_2 = cv2.VideoCapture(r'star_hole_y/star_insertion_break_2/flow_init picture/flow_init_2.avi')

    Flow = Flow(col=320, row=240)

    col = 320
    row = 240
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

    out = cv2.VideoWriter('flow picture/flow.avi', fourcc, 3.5, (col * 1, row * 1))  # The fps depends on CPU
    out_2 = cv2.VideoWriter('flow_init picture/flow_init.avi', fourcc, 3.5, (col * 1, row * 1))
    out_3 = cv2.VideoWriter('flow picture/flow_2.avi', fourcc, 3.5, (col * 1, row * 1))  # The fps depends on CPU
    out_4 = cv2.VideoWriter('flow_init picture/flow_init_2.avi', fourcc, 3.5, (col * 1, row * 1))

    time.sleep(1)
    # 第一个传感器
    ret, frame1 = cap.read()
    # frame1 = cv2.resize(frame1, (col, row))
    frame1 = Flow.get_raw_img(frame1)
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    f0 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # 第二个传感器
    ret, frame1_2 = cap_2.read()
    # frame1_2 = cv2.resize(frame1_2, (col, row))
    frame1_2 = Flow.get_raw_img(frame1_2)
    prvs_2 = cv2.cvtColor(frame1_2, cv2.COLOR_BGR2GRAY)
    f0_2 = cv2.cvtColor(frame1_2, cv2.COLOR_BGR2GRAY)

    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    flow_sum = np.zeros((Flow.row, Flow.col, 2))
    flow_trend = np.zeros(0)
    count = 0
    d = {}

    reset_threshold_error = 0.3  # ?
    reset_threshold_mean = 2  # ?

    cluster_labels_new = np.zeros(0)
    weights_new = np.zeros(0)
    Y_pred_list = []

    # 创建一个队列，用于在主进程和子进程之间传递数据
    data_queue = multiprocessing.Queue()

    # 创建子进程1
    send_process = multiprocessing.Process(target=send_mode_transfer, args=(data_queue,))
    send_process.start()
    while (1):
        count += 1
        # 第一个传感器
        try:
            ret, frame2 = cap.read()
            frame2 = Flow.get_raw_img(frame2)
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            t1 = time.time()
        except:
            break

        flow = cv2.calcOpticalFlowFarneback(f0, next, None, 0.5, 3, int(180 * Flow.col / 960), 5, 5, 1.2, 0)

        flow_sum[:, :, 0] = flow[:, :, 0]
        flow_sum[:, :, 1] = flow[:, :, 1]

        frame3 = np.copy(frame2)
        d_flow_1 = Flow.draw(frame3, flow_sum)  # frame3是此刻与初始时刻的光流图（校正后）
        cv2.imshow('frame', frame3)

        k = cv2.waitKey(30) & 0xff
        out.write(frame3)
        out_2.write(frame2)
        # cv2.imwrite(r'flow_init picture/flow_init' + str(count) + '.png', frame2)
        # cv2.imwrite(r'flow picture/flow' + str(count) + '.png', frame3)
        if k == 27:  # K=ESC
            break
        prvs = next  #

        # 第二个传感器
        try:
            ret, frame2 = cap_2.read()
            frame2 = Flow.get_raw_img(frame2)
            next_2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        except:
            break

        flow = cv2.calcOpticalFlowFarneback(f0_2, next_2, None, 0.5, 3, int(180 * Flow.col / 960), 5, 5, 1.2, 0)
        flow_sum[:, :, 0] = flow[:, :, 0]
        flow_sum[:, :, 1] = flow[:, :, 1]

        frame3 = np.copy(frame2)
        d_flow_2 = Flow.draw(frame3, flow_sum)  # frame3是此刻与初始时刻的光流图（校正后）

        d_flow = np.append(d_flow_1, d_flow_2)

        cv2.imshow('frame2', frame3)

        k = cv2.waitKey(30) & 0xff
        out_3.write(frame3)
        out_4.write(frame2)
        if k == 27:  # K=ESC
            break
        prvs_2 = next_2  #

        t2 = time.time()
        print('freq=', 1.0 / (t2 - t1))
        if k == ord("r"):  # to make a reset of the flow
            # f0_2 = next_2
            f0 = next
            f0_2 = next_2

        d_flow = d_flow.astype(np.float32)
        d_flow = (d_flow - mean) / std
        d_flow = d_flow.reshape(1, -1)
        weights_now = pca.transform(d_flow)
        weights_new = np.append(weights_new, weights_now).reshape(-1, 5)
        cluster_labels_now = gmm_model.predict(weights_now)
        cluster_labels_new = np.append(cluster_labels_new, cluster_labels_now)
        cluster_labels_new = cluster_labels_new.astype(np.int32)  # 这一步很重要，因为之前训练时的标签是整数型，故这一步要统一成整数型
        X_features_new = [[word2features(cluster_labels_new, i) for i in range(len(cluster_labels_new))]]
        Y_pred_new = crf_model.predict(X_features_new)
        Y_pred_new_now = Y_pred_new[0][-1]
        Y_pred_list.append(Y_pred_new_now)
        print('the mode now=', Y_pred_new_now)

        message = 'r' + Y_pred_new_now + '\n'
        # 每当 Y_pred_new_now 产生新值时，将其放入队列
        data_queue.put(message)

    cap.release()
    cv2.destroyAllWindows()
    send_process.terminate()

# with open('first time/star_hole_x_1.txt', 'w') as f:
#     for item in Y_pred_list:
#         f.write("%s\n" % item)