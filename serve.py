# -*-coding: utf-8 -*-
#引入socket库
import socket
import time
import cv2
import numpy as np

# 接受图片大小的信息
def recv_size_start(sock, count):
     buf = ''
     while count:
         newbuf = sock.recv(count)
         newbuf_1 = newbuf.decode('utf-8')
         if not newbuf_1: return None
         buf += newbuf_1
         count -= len(newbuf_1)
     return buf


def recv_size(sock, count):
    buf = b""
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


#这是进行定义一个ip协议版本AF_INET（IPv4），定义一个传输TCP协议，SOCK_STREAM
sk=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
#定义ip地址与端口号，ip地址就是服务端的ip地址，端口随便定义，但要与客户端脚本一致
ip_port=('localhost', 30001)#进行连接服务器
#绑定一个端口
sk.bind(ip_port)
#监听一个端口,这里的数字3是一个常量，表示阻塞3个连接，也就是最大等待数为3
sk.listen(3)
#接受客户端的数据，并返回两个参数，a为连接信息，b为客户端的ip地址与端口号
a,b=sk.accept()
print(a)
t1 = time.time()
while True:
    # #接收str数据
    # data=a.recv(1024)#客户端发送的数据存储在recv里，1024指最大接受数据的量
    # print(data.decode('utf-8'))
    # message = data.decode('utf-8')
    # message="you can say:"
    # a.send(message.encode('utf-8'))
    # t2 = time.time()
    # if t2 - t1 !=0:
    #     print('freq=', 1 / (t2 - t1))
    # t1 = t2
    # if message==('bye'):
    #     break

    #接收视频流（图片）数据
    length = recv_size_start(a, 16)  # 首先接收来自客户端发送的大小信息
    if isinstance(length, str):  # 若成功接收到大小信息，进一步再接收整张图片
        stringData = recv_size(a, int(length))
        data = np.fromstring(stringData, dtype='uint8')
        decimg = cv2.imdecode(data, 1)  # 解码处理，返回mat图片
        cv2.imshow('SERVER', decimg)
        if cv2.waitKey(10) == 27:
            break
        print('Image recieved successfully!')
    if cv2.waitKey(10) == 27:
        break