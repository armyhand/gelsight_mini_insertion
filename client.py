# -*- coding:utf-8 -*-
#导入socket库
import socket
import time
import numpy as np
import struct

def send_mode_transfer(queue):
    prev_mode_transfer = '12345'  # 用于保存上一个时刻的 Y_pred_new_now
    # 定义一个ip协议版本AF_INET，为IPv4；同时也定义一个传输协议（TCP）SOCK_STREAM
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 定义IP地址与端口号
    ip_port = ('172.31.1.147', 30001)  # 进行连接服务器
    # ip_port = ('localhost', 30001)  # 进行连接服务器
    client.connect(ip_port)
    print('success connect!')
    while True:
        # 从队列获取数据
        new_mode_transfer = queue.get()

        # 如果新数据不为空，则更新 prev_mode_transfer
        if new_mode_transfer is not None:
            prev_mode_transfer = new_mode_transfer

        # 在这里添加与服务端通信的代码，将 prev_mode_transfer 发送给服务端并接收返回的信息
        if prev_mode_transfer is not None:
            # 发送 prev_mode_transfer 给服务器
            client.sendall(prev_mode_transfer.encode('utf-8'))

            received_data = client.recv(1024)  # 接受服务端的信息，最大数据为1k
            if len(received_data) == 144:
                floats_bytes = received_data
                B.append(received_data)
                float_count = len(floats_bytes) // 4
                floats = struct.unpack('>' + 'f' * float_count, floats_bytes)
                # shared_data = floats
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
                floats = struct.unpack('>' + 'f' * float_count, floats_bytes)
                # shared_data = floats


def rece_mode_transfer(shared_data):
    # 这是进行定义一个ip协议版本AF_INET（IPv4），定义一个传输TCP协议，SOCK_STREAM
    sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 定义ip地址与端口号，ip地址就是服务端的ip地址，端口随便定义，但要与客户端脚本一致
    ip_port = ('172.31.1.147', 30001)  # 进行连接服务器
    # 绑定一个端口
    sk.bind(ip_port)
    # 监听一个端口,这里的数字3是一个常量，表示阻塞3个连接，也就是最大等待数为3
    sk.listen(3)
    # 接受客户端的数据，并返回两个参数，a为连接信息，b为客户端的ip地址与端口号
    conn, addr = sk.accept()
    with conn:
        print(f"Connected by {addr}")
        while True:
            # # 接收服务器返回的信息
            received_data = sk.recv(1024)  # 接受服务端的信息，最大数据为1k
            if len(received_data) == 144:
                floats_bytes = received_data
                B.append(received_data)
                float_count = len(floats_bytes) // 4
                floats = struct.unpack('>' + 'f' * float_count, floats_bytes)
                shared_data = floats
                A.append(floats)
                # print(floats)
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
                floats = struct.unpack('>' + 'f' * float_count, floats_bytes)
                shared_data = floats


#定义一个ip协议版本AF_INET，为IPv4；同时也定义一个传输协议（TCP）SOCK_STREAM
client=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
#定义IP地址与端口号
ip_port=('172.31.1.147', 30001)
#进行连接服务器
client.connect(ip_port)
A = []
B = []
while True:
    # message=input('You can say:')
    # client.send(message.encode('utf-8'))#将发送的数据进行编码
    message_float = [3.14, 6.28, 9.42]
    message_str = 'r' + str(message_float) + '\n'
    client.sendall(message_str.encode('utf-8'))

    received_data = client.recv(1024)#接受服务端的信息，最大数据为1k
    if len(received_data) == 144:
        floats_bytes = received_data
        B.append(received_data)
        float_count = len(floats_bytes) // 4
        floats = struct.unpack('>' + 'f' * float_count, floats_bytes)
        A.append(floats)
        print(floats)
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
        floats = struct.unpack('>' + 'f' * float_count, floats_bytes)
        A.append(floats)
        print(floats)
# if len(A) > 100:
#     np.savetxt(r'force_bytes.txt', A)
client.close()
