import numpy as np
import math

# 获取物体的位姿
X = 1.0  # X坐标
Y = 2.0  # Y坐标
Z = 3.0  # Z坐标
Roll = math.radians(30)  # 绕X轴的旋转角度，将角度转换为弧度
Pitch = math.radians(45)  # 绕Y轴的旋转角度，将角度转换为弧度
Yaw = math.radians(60)  # 绕Z轴的旋转角度，将角度转换为弧度

# 计算旋转矩阵R（绕Z轴→绕Y轴→绕X轴）
Rz = np.array([[np.cos(Yaw), -np.sin(Yaw), 0],
               [np.sin(Yaw), np.cos(Yaw), 0],
               [0, 0, 1]])

Ry = np.array([[np.cos(Pitch), 0, np.sin(Pitch)],
               [0, 1, 0],
               [-np.sin(Pitch), 0, np.cos(Pitch)]])

Rx = np.array([[1, 0, 0],
               [0, np.cos(Roll), -np.sin(Roll)],
               [0, np.sin(Roll), np.cos(Roll)]])

R = Rz.dot(Ry).dot(Rx)  # 组合旋转矩阵

# 计算平移向量T
T = np.array([[X],
              [Y],
              [Z]])

# 打印结果
print("旋转矩阵R:")
print(R)
print("\n平移向量T:")
print(T)
