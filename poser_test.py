from utils.pressure_sensor import PressureSensor
import numpy as np
import threading
import time

# 本程序调用了调用is_touch_successful函数，返回当前的机械臂夹持状态和max_values

def eval(pressure_sensor):
    while True:
            result = pressure_sensor.is_touch_successful()
            print(result)
            time.sleep(1)

if __name__ == '__main__':
    # 创建并启动数据接收线程
    pressure_sensor = PressureSensor()
    receive_thread = threading.Thread(target=pressure_sensor.receive_data)
    receive_thread.start()
    time.sleep(1)
    eval(pressure_sensor)