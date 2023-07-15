# MPV_2024
假肢高层控制程序
## LCM 通信
上层规划和`ros_ctrl`节点采用LCM通信，消息格式如下
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
更改消息格式后进入`lcm_msg`文件夹，运行`lcm_msg_update.sh`文件更新消息格式，将对应文件夹下的`.hpp`文件复制到`ros_ctrl`功能包下

更多详细内容参考`Pros_Ctrl`