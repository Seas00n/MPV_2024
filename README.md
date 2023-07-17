# MPV_2024 假肢高层逻辑

## 控制器
### 动力学辨识
### 控制器

## 步态数据库
### 凸优化
### GMM

## 决策器
### 环境参数调节


## 传感器
### IMU:IM948
9轴蓝牙通信，参考仓库 https://github.com/Seas00n/IM948
### Tof Camera: Flex2
每秒15帧，LCM通信实测不丢包，参考仓库 https://github.com/Seas00n/RoyaleFlex2_CPP


## LCM 通信
上层规划和`ros_ctrl`节点采用LCM通信，消息格式如下

上层发送指令格式
```angular2html
package mvp_t;

struct msg_t
{
    float knee_position_desired;
    float ankle_position_desired;
    float knee_velocity_desired;
    float ankle_velocity_desired;
    float knee_torque_desired;
    float ankle_torque_desired;
    byte Mode;
    float Kp;
    float Kd;
    float Angle_eq;
}
```
其中`Mode`状态如下
```angular2html
CMD_POSITION_CTRL=0,
CMD_VELOCITY_CTRL=1,
CMD_TORQUE_CTRL=2,
CMD_POSITION_AND_VELOCITY=3,
CMD_IMPEDANCE=4,
CMD_QUICK_STOP=5
```
下层传递消息格式
```angular2html
package mvp_r;
struct msg_r
{
    float knee_position_actual;
    float ankle_position_actual;
    float knee_velocity_actual;
    float ankle_velocity_actual;
    float knee_torque_actual;
    float ankle_torque_actual;
    byte Error_Code;
    float knee_temperature;
    float ankle_temperature;
}
```
其中`Error_Code`为`0`时为正常

更改消息格式后进入`lcm_msg`文件夹，运行`lcm_msg_update.sh`文件更新消息格式，将对应文件夹下的`.hpp`文件复制到`ros_ctrl`功能包下

更多详细内容参考`Pros_Ctrl`仓库