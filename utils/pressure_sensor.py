import socket
import struct
import threading
import numpy as np
import queue
# from draw0 import draw
import matplotlib.pyplot as plt
import os
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter
import time

# 本代码是用来判断传感器是否夹住物体的代码,可以直接调用is_touch_successful()函数来判断是否夹住物体


class PressureSensor:
    def __init__(self, server_ip='192.168.31.222', server_port=888, keyword=b'touch', packet_size=2872, expected_datalength=2856):
        self.server_ip = server_ip
        self.server_port = server_port
        self.keyword = keyword
        self.packet_size = packet_size
        self.expected_datalength = expected_datalength
        # self.data_queue = queue.Queue()
        self.Z1 = None
        self.DIFF = 400
        self.FILTER_WINDOW_SIZE = 3
        self.result = None
        self.stop_event = threading.Event() 


    def gaussian_smoothing(self, data, sigma=1):
        """使用高斯滤波平滑数据"""
        return gaussian_filter(data, sigma=sigma)


    def receive_data(self):
        global Z1
        # 配置服务器的 IP 地址和端口号
        SERVER_IP = '192.168.31.222'  # 请替换为实际的服务器 IP
        SERVER_PORT = 888           # 请替换为实际的服务器端口

        # 定义 keyworld 和数据包大小
        KEYWORLD = b'touch'           # 数据包头标识符
        PACKET_SIZE = 2872            # 整个数据包的字节数
        EXPECTED_DATALENGTH = 2856    # 期望的 datalength 值

        # 创建一个空的缓冲区
        buffer = bytearray()
        data_count = 0
        Z1 = None
        cnt=0

        try:
            # 创建一个 TCP/IP 套接字
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                # 连接到服务器
                print(f'正在连接到 {SERVER_IP}:{SERVER_PORT}...')
                sock.connect((SERVER_IP, SERVER_PORT))
                print('连接成功！开始接收数据...')
                

                while not self.stop_event.is_set() and data_count < 800:

                    
                    # 从套接字接收数据
                    data = sock.recv(4096)
                    if not data:
                        print('连接关闭。')
                        break
                    # 将接收到的数据添加到缓冲区
                    buffer.extend(data)

                    # 处理缓冲区中的数据
                    while True:

                        # 如果缓冲区中的数据少于 keyworld 的长度，继续接收数据
                        if len(buffer) < len(KEYWORLD):
                            break

                        # 检查缓冲区的开头是否为 keyworld
                        if buffer[:len(KEYWORLD)] != KEYWORLD:
                            # 如果不是，查找下一个可能的 keyworld 开始位置
                            pos = buffer.find(KEYWORLD, 1)
                            if pos == -1:
                                # 如果找不到，清空缓冲区中的前 len(KEYWORLD)-1 个字节
                                buffer = buffer[-(len(KEYWORLD)-1):]
                                break
                            else:
                                # 移除找到的位置之前的所有字节
                                buffer = buffer[pos:]
                        
                        # 检查缓冲区是否有足够的数据包
                        if len(buffer) < PACKET_SIZE:
                            break

                        # 提取一个完整的数据包
                        packet = buffer[:PACKET_SIZE]

                        # 解析 datalength 字段（第 6-7 字节，unsigned short，Little Endian）
                        datalength = struct.unpack('<H', packet[5:7])[0]

                        # 验证 datalength 是否正确
                        if datalength != EXPECTED_DATALENGTH:
                            print(f'无效的数据包长度: {datalength}，预期: {EXPECTED_DATALENGTH}')
                            # 移除第一个字节，继续查找下一个数据包
                            buffer.pop(0)
                            continue

                        # 解析 TouchSensorData（从第 16 字节开始，352x4 个 int16）
                        touch_start = 16
                        touch_end = touch_start + (352 * 4 * 2)  # 每个 int16 2 字节
                        touch_data = packet[touch_start:touch_end]

                        # 使用 struct 解包为 352*4 个 int16
                        touch_format = '<' + 'h' * (352 * 4)
                        try:
                            touch_values = struct.unpack(touch_format, touch_data)
                        except struct.error as e:
                            print(f'解包 TouchSensorData 失败: {e}')
                            # 移除一个字节，尝试下一个可能的数据包
                            buffer.pop(0)
                            continue

                        # 将触摸传感器数据划分为 352x4 的二维数组
                        touch_array_base = [touch_values[i*4:(i+1)*4] for i in range(352)]
                        touch_array=[touch_array_base[i*16:(i+1)*16] for i in range(22)]
                        

                        # 指定输出某一组，例如输出第一组的 16 个值
                        index_to_output = 0  # 指定要输出的组索引
                        if index_to_output < len(touch_array):
                            output_values = touch_array[index_to_output]  # 取出指定组的值
                            

                        # all_x=[]
                        # all_y=[]
                        all_z=[]
                        # all_t=[]
                        for i in range(16):
                            # x = output_values[i][0]/abs(output_values[i][0])*int(output_values[i][0]- 32768)/32768.0*255*5  # 获取 x 值
                            # y = int(output_values[i][1]- 32768)/32768.0*255*8  # 获取 y 值
                            # z = int(output_values[i][2]- 32768)/32768.0*255  # 获取 z 值
                            # x = round(output_values[i][0]/abs(output_values[i][0])*(abs(output_values[i][0]) -22000)/15000, 5)  # 获取 x 值
                            # y = round(output_values[i][1]/abs(output_values[i][1])*(abs(output_values[i][1])-25000)/15000, 5)  # 获取 y 值
                            # z = round(output_values[i][2]/abs(output_values[i][2])*(abs(output_values[i][2])-27000)/8000, 5)  # 获取 z 值
                            z = round(np.abs(output_values[i][2]), 5)  # 获取 z 值

                            # 将数据添加到相应的列表中
                            # all_x.append(x)
                            # all_y.append(y)
                            all_z.append(z)
                        

                        
                        # 将z列表转换为z的数组
                        z_array = np.array([all_z])
                        cnt+=1
                        # 如果 Z1 为空，则记录第一组 z 数据
                        if Z1 is None:
                            Z1 = z_array
                            # print(f"Recorded first z data: {Z1}")
                            continue  # 跳过本次循环，不将 Z1 放入队列
                        else:
                            # 后续的所有数组减去第一组保存下来的 Z1
                            z_array -= Z1
                            z_array = np.abs(z_array)  # 取绝对值
                        
                    
                        # print(f"raw_z_array: {z_array}")
                        # # 应用高斯滤波进行平滑
                        z_array_smoothed = self.gaussian_smoothing(z_array, sigma=1)
                        
                        z_array = np.delete(z_array_smoothed, [0, -1])
                        # print(f"smoothed_z_array: {z_array}")


                        # # 中位数滤波
                        # z_array_smoothed = medfilt(z_array, kernel_size=FILTER_WINDOW_SIZE)
                        # z_array = np.round(z_array_smoothed, 5)  # 保留 5 位小数     

                        #  # 应用移动平均进行平滑
                        # z_array_smoothed = moving_average(z_array.flatten(), FILTER_WINDOW_SIZE)
                        # z_array = np.round(z_array_smoothed, 5)  # 保留 5 位小数 
                        # 计算每一维的最大值和最小值
                        # 确保 Z1 已经被赋值
                        
                        if Z1 is not None:
                        
                            # 计算每一维的最大值和最小值
                            max_values = np.max(z_array)
                            min_values = np.min(z_array)
                            # self.data_queue.put(z_array) # 将数据添加到全局队列中
                            if cnt == 800:  # 只在第500个数据时进行判断
                                if max_values > self.DIFF:
                                    print(f"Max values: {max_values}")
                                    # print(z_array)
                                    print("successsfully touch")
                                    print(time.time())
                                    self.result = {"success": max_values}
                                else:
                                    print("no touch")
                                    print(time.time())
                                    self.result = {"no_touch": max_values}
                                self.stop_event.set()  # 设置停止事件，结束线程
                                

                        # print(f"Max values: {max_values}")
                        # print(f"Min values: {min_values}")

                        # data_queue.put(z_array) # 将数据添加到全局队列中
                        # 打印 z 的数组
                        # print(f"Smoothed z data: {z_array}")
        
                        # # 输出所有的 x 值，格式化为每个字符宽度为 10
                        # output_x = ' '.join(f'{value:<8}' for value in all_x)  # 使用左对齐格式化
                        # print('X: ',output_x)
                        # output_y = ' '.join(f'{value:<8}' for value in all_y)  # 使用左对齐格式化
                        # print('Y: ',output_y)
                        # output_z = ' '.join(f'{value:<8}' for value in all_z)  # 使用左对齐格式化
                        # print('Z: ',output_z)
                        # print('-----------------------\n')

                        # 移除已处理的数据包
                        buffer = buffer[PACKET_SIZE:]
                        # if cnt ==799:
                        #     print("接收数据完成，主动结束。")
                        #     return  # 退出函数，主动结束
                 

        except ConnectionRefusedError:
            print(f'无法连接到 {SERVER_IP}:{SERVER_PORT}。请检查服务器是否在运行。')
        except KeyboardInterrupt:
            print('\n程序终止。')
        except Exception as e:
            print(f'发生错误: {e}')

    def run_and_get_result(self):
        """运行接收数据线程并获取结果"""
        receive_thread = threading.Thread(target=self.receive_data)
        receive_thread.start()
        receive_thread.join()  # 等待接收线程结束
        return self.result




def main():
    # 创建并启动数据接收线程
    pressure_sensor = PressureSensor()
    result = pressure_sensor.run_and_get_result()
    print(result)

    # 启动绘图
    #plt.show()

if __name__ == '__main__':
    main()