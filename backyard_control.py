import socket
import sys
import time

HOST, PORT = "192.168.137.9", 9999 #夹爪 IP 和 TCP/IP 服务器端口号
#定义 moveTo 函数
def moveTo(sock,position,speed,acceleration,torque,tolerance=90,waitFlag=True):
    cmd="moveTo"
    cmd_args=cmd+"({0},{1},{2},{3},{4},{5})".format(position,speed,acceleration,torque,tolerance,waitFlag)+"\n"
    sock.sendall(bytes(cmd_args,"utf-8"))
    return int(str(sock.recv(1024), "utf-8"))

#定义夹爪标定函数
def calibrateGripper(sock):
    cmd_args="calibrateGripper()"+"\n"
    sock.sendall(bytes(cmd_args,"utf-8"))
    return int(str(sock.recv(1024), "utf-8"))
#获取夹爪当前状态
def getStatus(sock):
    cmd_args="getStatus()"+"\n"
    sock.sendall(bytes(cmd_args,"utf-8"))
    return str(sock.recv(1024), "utf-8")
#获取夹爪是否已经标定
def getCalibrated(sock):
    cmd_args="getCalibrated()"+"\n"
    sock.sendall(bytes(cmd_args,"utf-8"))
    return str(sock.recv(1024), "utf-8")
#关闭夹爪控制器
def shutdown(sock):
    cmd_args="shutdown()"+"\n"
    sock.sendall(bytes(cmd_args,"utf-8"))
    return str(sock.recv(1024), "utf-8")
#重启夹爪控制器
def restart(sock):
    cmd_args="restart()"+"\n"
    sock.sendall(bytes(cmd_args,"utf-8"))
    return str(sock.recv(1024), "utf-8")
#创建 socket 对象
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
#连接夹爪
    sock.connect((HOST, PORT))
    if(calibrateGripper(sock)==0): #返回 0 则，标定成功
        print(getCalibrated(sock))
        print(moveTo(sock,90,150,700,0.5)) #运动到 60mm 位置
        print(getStatus(sock)) #获取当前状态
        time.sleep(3)
        print(moveTo(sock,10,150,700,0.5)) #运动到 10mm 位置
        print(getStatus(sock))
        print(moveTo(sock,65,150,700,0.5))
        print(moveTo(sock,0,150,700,0.5))
except Exception as e:
    print(e)

finally:
    sock.close() #关闭 socket 释放资源