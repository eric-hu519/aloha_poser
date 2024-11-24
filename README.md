# A CoP implementation of Aloha Robot
## 现有功能：
1.单机械臂及双机械臂切换
2.双边机械臂标定及运动
3.两个以内目标物体检测并自动分配邻近机械臂
4.根据机械臂夹持状态及目标检测状态自动改变夹持位姿
5.通过LLM实现单边机械臂位置、俯仰角及偏转角控制
6.单独的机械臂调试功能
7.由于右机械臂上含有触觉传感器，现在夹爪闭合程度会根据左右机械臂自动判断

## 待优化及完善：
1.多个物体检测自动分配机械臂功能完善
2.双臂协同同步问题完善
3.相机检测精准度提高
4.完善与LLM交互功能，更新双臂场景下的LLM交互
5.改进Prompt，提供更加丰富的双臂场景及复杂动作组合，加入side参数
6.加入碰撞检测或者急停功能，增加机械臂使用安全性

## 主要文件：
1.`robot_controller.py`：机械臂控制器，与动作序列数据交互，包含动作序列解析、机械臂动作api、视觉检测api、单/双机械臂动作执行等功能，可以通过提供包含动作序列的json文件直接控制机械臂运动，便于调试。
2.`aloha_poser.py`：提供与LLM交互功能，通过prompt指导LLM生成指定格式的动作序列，并交给机械臂控制器解析并执行。
3.`robot_position.py`：提供单边机械臂的调试功能，可以通过键盘操控机械臂到达指定位置并记录对应状态下的坐标。也可使用该程序指导机械臂复位。
4.`HandEyeCalibration_eye-to-hand_realsense2_right.py`，`HandEyeCalibration_eye-to-hand_realsense2.py`：机械臂手眼标定程序，如果相机位置发生改动，则需要运行该程序重新对左右手进行标定，标定完成后，需要在`robot_controller.py`更新最新标定结果文件夹路径以加载相关参数。

