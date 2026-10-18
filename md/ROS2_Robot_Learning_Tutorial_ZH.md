# 机器人学习中的 ROS 2 完整教程

<!-- course-meta:start -->
[课程首页](https://github.com/zengxiong111/Foundation_of_Embodied_AI/blob/main/README.zh-CN.md) · [English](https://github.com/zengxiong111/Foundation_of_Embodied_AI/blob/main/md/ROS2_Robot_Learning_Tutorial_EN.md) · [PDF](https://zengxiong111.github.io/zengxiong.github.io/files/ROS2_Robot_Learning_Tutorial_ZH.pdf)

| 项目 | 说明 |
|---|---|
| 适用范围 | Ubuntu 24.04；ROS 2 Jazzy；Python 示例 |
| 文档核验日期 | 2026-10-18 |
| 许可 | CC BY 4.0 · Xiong Zeng and contributors |

**核验边界：** 文档整理与上游引用快照核对；未进行教程环境、训练或硬件验证。 固定源码引用见[资料清单](https://github.com/zengxiong111/Foundation_of_Embodied_AI/blob/main/materials.json)。原创教材采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，第三方引用、图片及代码遵循其原有许可；注明来源不等于拥有再许可权。
<!-- course-meta:end -->

<!-- course-toc:start -->
## 目录

- [一、ROS 是什么？为什么机器人学习需要 ROS？](#course-section-01)
  - [1.1 ROS 的定义](#course-section-02)
  - [1.2 一个典型的机器人学习系统](#course-section-03)
  - [1.3 机器人学习一定需要 ROS 吗？](#course-section-04)
- [二、ROS 1 与 ROS 2，以及发行版选择](#course-section-05)
  - [2.1 ROS 1 与 ROS 2 的主要区别](#course-section-06)
  - [2.2 2026 年该选哪个 ROS 版本？](#course-section-07)
- [三、ROS 2 的核心概念](#course-section-08)
  - [3.1 Node：节点](#course-section-09)
  - [3.2 Topic：发布 / 订阅](#course-section-10)
  - [3.3 Message：消息格式](#course-section-11)
  - [3.4 Service：请求 / 响应](#course-section-12)
  - [3.5 Action：需要进度反馈的长任务](#course-section-13)
  - [3.6 Parameter：节点配置](#course-section-14)
  - [3.7 TF2：机器人坐标系管理](#course-section-15)
  - [3.8 QoS：通信质量配置](#course-section-16)
- [四、Ubuntu 24.04 安装 ROS 2 Jazzy](#course-section-17)
  - [4.1 检查系统、添加软件源](#course-section-18)
  - [4.2 安装 ROS 2 和开发工具](#course-section-19)
  - [4.3 配置环境并验证](#course-section-20)
  - [4.4 第一个 ROS 通信测试](#course-section-21)
  - [4.5 必须掌握的 ROS 命令](#course-section-22)
- [五、实操：用 Python 编写机器人学习策略节点](#course-section-23)
  - [5.1 创建 Workspace](#course-section-24)
  - [5.2 创建 Python Package](#course-section-25)
  - [5.3 编写策略节点](#course-section-26)
  - [5.4 声明依赖和命令入口](#course-section-27)
  - [5.5 编译并运行](#course-section-28)
  - [5.6 模拟状态输入并验证输出](#course-section-29)
  - [5.7 替换成 PPO / SAC 网络推理](#course-section-30)
- [六、ROS 2 如何用于机器人数据采集？](#course-section-31)
  - [6.1 遥操作数据采集架构](#course-section-32)
  - [6.2 什么是 rosbag2？](#course-section-33)
  - [6.3 记录与回放数据](#course-section-34)
  - [6.4 对机器人学习有用的数据字段](#course-section-35)
  - [6.5 多频率同步与数据集转换](#course-section-36)
- [七、URDF、TF2、RViz 和 ros2_control](#course-section-37)
  - [7.1 URDF：机器人结构描述](#course-section-38)
  - [7.2 URDF 与 TF2 的关系](#course-section-39)
  - [7.3 RViz 2：可视化工具](#course-section-40)
  - [7.4 ros2_control：连接高层命令与硬件](#course-section-41)
- [八、ROS 如何连接 Isaac Lab / MuJoCo 强化学习？](#course-section-42)
  - [8.1 仿真训练通常不需要 ROS](#course-section-43)
  - [8.2 真机部署需要连接各个模块](#course-section-44)
  - [8.3 Sim2Real 最关键的是接口语义一致](#course-section-45)
  - [8.4 控制频率与延迟](#course-section-46)
- [九、Tron2 + Revo3 灵巧操作项目架构](#course-section-47)
  - [9.1 推荐节点与信息流](#course-section-48)
  - [9.2 推荐 Workspace 组织](#course-section-49)
  - [9.3 机械臂与灵巧手的动作分发](#course-section-50)
  - [9.4 FoundationPose 的 ROS 接入](#course-section-51)
- [十、机器人学习中的 ROS 常见问题](#course-section-52)
  - [10.1 ROS 2 与 Conda 的关系](#course-section-53)
  - [10.2 与真实机器人相关的安全边界](#course-section-54)
- [十一、推荐学习路线](#course-section-55)
- [十二、总结与实践建议](#course-section-56)
- [十三、官方文档与参考资料](#course-section-57)
<!-- course-toc:end -->

**章节概念图**

```mermaid
flowchart LR
  A["传感器"] --> B["ROS Topic"] --> C["策略节点"] --> D["控制器"]
```


<a id="course-section-01"></a>

## 一、ROS 是什么？为什么机器人学习需要 ROS？

<a id="course-section-02"></a>

### 1.1 ROS 的定义

ROS（Robot Operating System）虽然叫“机器人操作系统”，但并不是 Ubuntu、Windows 这样的传统操作系统。它是机器人软件开发框架及其生态系统，核心作用是让机器人中的不同软件模块通信、共享数据并协作运行。

一套机器人通常包含相机、机械臂、灵巧手、物体位姿估计器、神经网络策略和底层控制器。它们可能用不同语言开发、运行在不同进程中，甚至分布在不同计算机上。ROS 通过统一的消息、通信和工具链连接这些组件。

<a id="course-section-03"></a>

### 1.2 一个典型的机器人学习系统

假设 Tron2 + Revo3 需要完成“拿起螺丝刀并旋转”的任务，闭环结构如下：

```text
                 真实机器人
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
     RGB-D / RGB 相机      机械臂 + 灵巧手
          │                     │
          ▼                     ▼
       图像数据                关节状态
          └──────────┬──────────┘
                     ▼
                  ROS 2
                     ▼
           Observation Builder
                     ▼
             RL Policy
              (PPO/SAC)
                     ▼
                  Action
                     ▼
                 ROS 2
                     ▼
             底层控制器 / SDK
                     ▼
            机械臂 + 灵巧手执行
                     └──→ 下一次观测
```

必须区分以下工具的职责：

| 组件 | 在机器人学习中的主要作用 |
|---|---|
| **ROS 2** | 模块通信、系统集成、坐标变换、数据记录与调试 |
| **PyTorch** | 训练、加载和推理神经网络 |
| **PPO / SAC** | 强化学习算法 |
| **Isaac Lab / MuJoCo** | 物理仿真与强化学习训练环境 |
| **ros2_control** | 硬件抽象、控制器管理及命令接口 |
| **MoveIt 2** | 运动规划、运动学和碰撞检查 |
| **RViz 2** | 机器人状态、传感器及 TF 可视化 |
| **rosbag2** | 消息记录和回放 |

**结论：ROS 负责“集成与通信”，学习框架负责“训练与推理”，控制器负责“执行”。** 三者互补，而非相互替代。

<a id="course-section-04"></a>

### 1.3 机器人学习一定需要 ROS 吗？

不一定。训练任务如果只需要 Python 和物理仿真器，ROS 并非必需。

| 场景 | 是否需要 ROS | 原因 |
|---|---|---|
| Isaac Lab 中并行训练 PPO | 不需要 | 直接调用仿真 API，减少额外开销 |
| MuJoCo 中训练 SAC | 不需要 | 环境与算法可以在 Python 内交互 |
| 离线训练 Diffusion Policy 或 VLA | 不需要 | 直接使用数据加载器和 PyTorch |
| 使用机器人厂商 SDK 控制单台机械臂 | 不一定 | 可以直接调用 SDK |
| 多相机 + 机械臂 + 灵巧手集成 | 推荐 | 方便模块解耦、监控和同步 |
| 遥操作数据采集 | 推荐 | 方便多模态数据记录与回放 |
| 强化学习策略真机部署 | 推荐但非必需 | 规范 Observation/Action 与设备接口 |
| 多机器人或分布式系统 | 推荐 | 便于跨进程、跨计算机通信 |

> **实用判断**：高吞吐量的仿真训练环路通常不放进 ROS；真实机器人的传感器、状态估计、控制接口、数据记录和部署，则适合使用 ROS 2。

---

<a id="course-section-05"></a>

## 二、ROS 1 与 ROS 2，以及发行版选择

<a id="course-section-06"></a>

### 2.1 ROS 1 与 ROS 2 的主要区别

ROS 1 是早期生态，ROS 2 面向现代机器人系统重新设计，在分布式发现、QoS、生命周期管理、跨平台和实时控制集成方面提供了更完整的基础设施。

| 特性 | ROS 1 | ROS 2 |
|---|---|---|
| 发现机制 | 通常依赖 ROS Master | 常见 DDS 中间件支持分布式发现 |
| Python 接口 | `rospy` | `rclpy` |
| C++ 接口 | `roscpp` | `rclcpp` |
| 记录工具 | `rosbag` | `rosbag2` |
| 参数 | ROS Parameter Server | 由节点管理参数 |
| QoS | 较有限 | 支持可靠性、历史、深度、期限等 |
| 控制框架 | `ros_control` | `ros2_control` |
| 新项目 | 一般不推荐从 ROS 1 开始 | 推荐使用 ROS 2 |

需要澄清：ROS 2 并非所有情况下都“自动实时”；它提供有利于实时系统集成的机制，真正的确定性仍取决于操作系统、控制线程、驱动和部署设计。

<a id="course-section-07"></a>

### 2.2 2026 年该选哪个 ROS 版本？

截至 **2026 年 10 月**，ROS 2 **Lyrical Luth** 已于 2026 年 5 月 22 日发布；但应当按当前 Ubuntu 系统和机器人驱动的兼容性选型。

| Ubuntu | 对应的常用 ROS 2 版本 | 备注 |
|---|---|---|
| Ubuntu 22.04 | Humble Hawksbill | 长期支持至 2027 年 5 月 |
| Ubuntu 24.04 | **Jazzy Jalisco** | 长期支持至 2029 年 5 月 |
| Ubuntu 26.04 | Lyrical Luth | 长期支持至 2031 年 5 月 |

**本教程使用 Ubuntu 24.04 + ROS 2 Jazzy。** 如果厂商驱动只支持指定发行版，必须优先核对驱动兼容性。

---

<a id="course-section-08"></a>

## 三、ROS 2 的核心概念

<a id="course-section-09"></a>

### 3.1 Node：节点

Node 是 ROS 的基本计算单元，可以理解为一个独立的软件功能模块。节点并不必然等于独立进程；多个节点也可以被组合在同一进程中。

```text
ROS 2 System
├── camera_node          # 读取图像
├── robot_state_node     # 读取关节状态
├── object_pose_node     # 估计物体 6D 位姿
├── rl_policy_node       # 策略推理
└── controller_node      # 执行动作
```

Node 分工清晰，可以独立测试与替换，例如只更新 FoundationPose 而不修改机械臂驱动。

<a id="course-section-10"></a>

### 3.2 Topic：发布 / 订阅

Topic 适合连续、异步数据流：

```text
camera_node --Publish--> /camera/image_raw --Subscribe--> perception_node
```

发送方无需知道接收方身份，一个 Topic 可以有多个订阅者。常见的机器人学习 Topic：

| Topic 示例 | 内容 |
|---|---|
| `/camera/color/image_raw` | RGB 图像 |
| `/camera/depth/image_rect_raw` | 深度图像 |
| `/joint_states` | 机器人关节状态 |
| `/object_pose` | 物体位姿 |
| `/policy/action` | 策略输出动作 |
| `/tf`、`/tf_static` | 动态、静态坐标变换 |

**Topic 的核心用途就是在感知、Observation 构建、Policy 和控制器之间传输数据。** 实际 Topic 名取决于驱动及命名空间。

<a id="course-section-11"></a>

### 3.3 Message：消息格式

Message 规定每条数据的字段与类型。常见类型：

| Message 类型 | 用途 |
|---|---|
| `sensor_msgs/msg/Image` | 图像 |
| `sensor_msgs/msg/CameraInfo` | 相机内参和标定相关信息 |
| `sensor_msgs/msg/JointState` | 关节名称、位置、速度、作用力 |
| `sensor_msgs/msg/Imu` | IMU 数据 |
| `geometry_msgs/msg/Pose` | 位置和姿态 |
| `geometry_msgs/msg/PoseStamped` | 带时间戳、坐标系的位姿 |
| `geometry_msgs/msg/Twist` | 线速度和角速度 |
| `trajectory_msgs/msg/JointTrajectory` | 关节轨迹 |
| `std_msgs/msg/Float64MultiArray` | 简单浮点数组 |

例如 `sensor_msgs/msg/JointState` 的主要字段：

```python
header       # 时间戳及头信息
name         # 关节名称，如 ['joint_1', 'joint_2']
position     # 关节位置
velocity     # 关节速度
effort       # 关节力矩或作用力
```

Revo3 是否直接发布 `JointState`、各关节的名字与顺序是什么，需要由实际驱动或包装层确定。**不要只凭数组下标假定不同设备的关节顺序一致。**

<a id="course-section-12"></a>

### 3.4 Service：请求 / 响应

Service 适合短时间的、需要响应的一次性操作：

```text
policy_node --Reset Request--> reset_server
policy_node <--Reset Result----- reset_server
```

典型操作有机器人 Reset、查询设备信息、切换控制模式、开始标定。Service 类似远程函数调用，但网络与服务端执行仍可能失败或超时。

<a id="course-section-13"></a>

### 3.5 Action：需要进度反馈的长任务

Action 适合“提交任务 → 反馈进度 → 返回最终结果”的过程，例如执行机械臂轨迹、导航到指定位置、完成抓取技能。Action 支持取消目标。

| 机制 | 最适合的通信模式 | 机器人学习例子 |
|---|---|---|
| Topic | 连续异步流 | 相机、状态、策略动作 |
| Service | 单次请求与响应 | Reset、查询、切换模式 |
| Action | 长任务、反馈、结果 | 执行一个抓取动作或轨迹 |

对于 50 Hz 之类的周期性闭环动作指令，通常采用 Topic 或直接进程内接口，而不是每个周期都新建一个 Action。

<a id="course-section-14"></a>

### 3.6 Parameter：节点配置

参数用于调整节点配置，不必修改主程序：

```yaml
policy_node:
  ros__parameters:
    control_frequency: 50.0
    use_camera: true
    observation_dim: 32
    policy_path: "/home/robot/policy.pt"
```

常见参数包括控制频率、相机名称、策略权重路径、目标坐标系、动作缩放与关节限位。

<a id="course-section-15"></a>

### 3.7 TF2：机器人坐标系管理

典型机器人包含多个坐标系：

```text
world
 └── base_link
      ├── arm_base
      │    └── ee_link
      │         └── hand_base
      └── head_link
           └── camera_link
                └── camera_optical_frame
```

例如 FoundationPose 计算物体在相机坐标系中的位姿 `T(C,O)`，抓取任务需要它在基座坐标系中的位姿：

**坐标变换：** `T(B,O) = T(B,C) × T(C,O)`

这里 `B` 表示机器人基座，`C` 表示相机，`O` 表示物体。TF2 允许查询相应坐标系之间、相应时间戳上的变换关系。

**关键注意：TF2 本身不负责相机标定。** 必须通过 URDF、手眼标定、外参测量等获得正确的 `T(B,C)`，然后 TF2 才能维护与使用这一关系。Tron2 头部相机、腕部相机、yaw/pitch 关节中心都涉及这一问题。

<a id="course-section-16"></a>

### 3.8 QoS：通信质量配置

QoS（Quality of Service）控制消息交付行为：

| 属性 | 含义 |
|---|---|
| Reliability | `Reliable` 或 `Best Effort` |
| History / Depth | 消息保留方式与队列长度 |
| Durability | 新订阅者是否可能收到历史保留数据 |
| Deadline | 期望更新期限；也可用于超时检测 |

传感器图像经常使用 `Best Effort`，避免追传过时帧；一些关键状态或控制指令可能使用 `Reliable`，但仍需避免旧指令积压。发布者和订阅者的 QoS 不兼容时，可能出现 Topic 存在却收不到数据的现象。

---

<a id="course-section-17"></a>

## 四、Ubuntu 24.04 安装 ROS 2 Jazzy

下面是基于官方 **Jazzy Ubuntu deb 包安装文档**的示例。安装命令需要联网、管理员权限，建议先确认机器尚未安装冲突的 ROS 发行版。

<a id="course-section-18"></a>

### 4.1 检查系统、添加软件源

```bash
lsb_release -a                 # 确认 Ubuntu 24.04
locale                         # 确认当前 locale 支持 UTF-8

sudo apt update
sudo apt install software-properties-common curl -y
sudo add-apt-repository universe

export ROS_APT_SOURCE_VERSION=$(curl -s \
  https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
  | grep -F 'tag_name' | awk -F\" '{print $4}')

curl -L -o /tmp/ros2-apt-source.deb \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"

sudo dpkg -i /tmp/ros2-apt-source.deb
```

如果系统没有 UTF-8 locale，可参考官方文档设置：

```bash
sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
```

<a id="course-section-19"></a>

### 4.2 安装 ROS 2 和开发工具

```bash
sudo apt update
sudo apt install ros-jazzy-desktop ros-dev-tools
```

`ros-jazzy-desktop` 包含 RViz、示例程序和常用工具。如果不需要 GUI，也可使用更精简的 `ros-jazzy-ros-base`。

<a id="course-section-20"></a>

### 4.3 配置环境并验证

```bash
source /opt/ros/jazzy/setup.bash
echo $ROS_DISTRO               # 应输出 jazzy
```

若希望 Bash 新终端自动加载：

```bash
echo 'source /opt/ros/jazzy/setup.bash' >> ~/.bashrc
```

不要在同一终端误加载多个 ROS 发行版。新开终端仍需要确认环境已正确 source。

<a id="course-section-21"></a>

### 4.4 第一个 ROS 通信测试

终端 A：

```bash
source /opt/ros/jazzy/setup.bash
ros2 run demo_nodes_cpp talker
```

终端 B：

```bash
source /opt/ros/jazzy/setup.bash
ros2 run demo_nodes_py listener
```

预期可以看到 `Publishing: Hello World` 与 `I heard: Hello World` 等输出，说明 Python 和 C++ 节点能够互相通信。

<a id="course-section-22"></a>

### 4.5 必须掌握的 ROS 命令

```bash
ros2 node list
ros2 node info /camera_node
ros2 topic list
ros2 topic echo /joint_states
ros2 topic type /joint_states
ros2 topic hz /joint_states
ros2 topic bw /camera/color/image_raw
ros2 topic info -v /joint_states
ros2 interface show sensor_msgs/msg/JointState
ros2 service list
ros2 action list
ros2 run tf2_ros tf2_echo base_link camera_link
```

以上节点、Topic 和 frame 名均为示例。真机调试时，优先掌握 `ros2 topic list`、`echo`、`hz`、`info -v`，以确认消息内容、频率与 QoS。

---

<a id="course-section-23"></a>

## 五、实操：用 Python 编写机器人学习策略节点

目标是构建一条不连接真实电机的、可复现的最小链路：

```text
/demo/joint_states
       │ (JointState)
       ▼
    Policy Node
       │ (Float64MultiArray)
       ▼
/demo/policy_target
```

<a id="course-section-24"></a>

### 5.1 创建 Workspace

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
```

Workspace 的常见目录：

```text
ros2_ws/
├── src/      # 源代码
├── build/    # 构建文件（编译时生成）
├── install/  # 安装后的程序（编译时生成）
└── log/      # 构建日志（编译时生成）
```

<a id="course-section-25"></a>

### 5.2 创建 Python Package

```bash
cd ~/ros2_ws/src
ros2 pkg create \
    --build-type ament_python \
    --license Apache-2.0 \
    --node-name demo_policy \
    robot_learning_demo
```

结构大致为：

```text
robot_learning_demo/
├── package.xml
├── setup.py
├── setup.cfg
├── resource/
└── robot_learning_demo/
    ├── __init__.py
    └── demo_policy.py
```

`package.xml` 定义依赖；`setup.py` 定义 Python 包安装与可执行命令入口；`demo_policy.py` 实现策略节点。

<a id="course-section-26"></a>

### 5.3 编写策略节点

编辑 `~/ros2_ws/src/robot_learning_demo/robot_learning_demo/demo_policy.py`：

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class PolicyNode(Node):
    def __init__(self):
        super().__init__("policy_node")

        # Subscribe: virtual robot observation
        self.subscription = self.create_subscription(
            JointState,
            "/demo/joint_states",
            self.state_callback,
            10,
        )

        # Publish: demonstration action (not for real hardware)
        self.publisher = self.create_publisher(
            Float64MultiArray,
            "/demo/policy_target",
            10,
        )

    def state_callback(self, msg):
        q = list(msg.position)
        if not q:
            return

        # An illustrative mapping, not a trained RL policy
        action = [0.5 * x for x in q]

        command = Float64MultiArray()
        command.data = action
        self.publisher.publish(command)


def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
```

这个演示策略为 `a_t = 0.5 q_t`，展示如何从 Observation 到 Action，不是安全的机器人控制策略。`Float64MultiArray` 也只用于快速演示；真实设备必须定义关节顺序、动作模式、单位、限位及命令超时处理。

<a id="course-section-27"></a>

### 5.4 声明依赖和命令入口

在 `package.xml` 内添加以下依赖（已有则不要重复）：

```xml
<exec_depend>rclpy</exec_depend>
<exec_depend>sensor_msgs</exec_depend>
<exec_depend>std_msgs</exec_depend>
```

确认 `setup.py` 的 `console_scripts` 包含：

```python
'demo_policy = robot_learning_demo.demo_policy:main'
```

<a id="course-section-28"></a>

### 5.5 编译并运行

```bash
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
ros2 run robot_learning_demo demo_policy
```

节点启动后会等待状态消息。注意 `colcon` 构建前的终端应已加载 Jazzy 环境。

<a id="course-section-29"></a>

### 5.6 模拟状态输入并验证输出

新终端，首先加载环境：

```bash
source /opt/ros/jazzy/setup.bash
```

发布两维虚拟关节状态，频率 5 Hz：

```bash
ros2 topic pub -r 5 \
  /demo/joint_states \
  sensor_msgs/msg/JointState \
  "{name: ['j1', 'j2'], position: [0.4, -0.2]}"
```

另一个终端：

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic echo /demo/policy_target
```

预期看到：

```text
data:
- 0.2
- -0.1
```

这样就完成了“发布状态 → 订阅状态 → 执行策略 → 发布动作”的最小闭环通信验证。

<a id="course-section-30"></a>

### 5.7 替换成 PPO / SAC 网络推理

训练完成后，可以把示例映射：

```python
action = [0.5 * x for x in q]
```

替换为 PyTorch 推理（示意代码）：

```python
with torch.inference_mode():
    action_tensor = policy(observation_tensor)
    action = action_tensor.cpu().numpy().tolist()
```

实际部署需要加载权重、构造正确 Observation、归一化、关节顺序映射、Action Scaling、处理设备和 dtype、限幅以及失败回退。

此外，更推荐把“消息接收”和“固定频率的推理周期”分开：

```text
Sensor Callback → Observation Buffer
                        │
                        ▼
               Policy Timer (例：50 Hz)
                        │
                        ▼
                Network Inference
                        │
                        ▼
               Safety Filter → Publisher
```

50 Hz 仅为示例，真实频率由模型推理时间、设备能力和任务决定。接收回调中无条件执行推理，可能引起频率漂移、消息积压和延迟。

---

<a id="course-section-31"></a>

## 六、ROS 2 如何用于机器人数据采集？

<a id="course-section-32"></a>

### 6.1 遥操作数据采集架构

以 PICO 遥操作 Tron2 + Revo3 为例：

```text
人类操作员 / PICO / Glove
          │
          ▼
    Teleoperation
          │
          ▼
 Robot Controller / SDK
          │
          ▼
     Real Robot
          │
    ┌─────┴────────────┐
    ▼                  ▼
RGB-D / RGB        Joint States
    │                  │
    └─────────┬────────┘
              ▼
            ROS 2
              ▼
           rosbag2
              ▼
        Offline Dataset
              ▼
BC / Diffusion Policy / VLA
```

<a id="course-section-33"></a>

### 6.2 什么是 rosbag2？

rosbag2 是 ROS 2 的消息记录与回放工具。它可以保存 Topic 消息及相关元数据，常见后端包括 SQLite3 和 MCAP。rosbag2 尤其适合重现数据采集过程，排查传感器和算法问题。

<a id="course-section-34"></a>

### 6.3 记录与回放数据

假设以下 Topic 已经存在：

```text
/joint_states
/camera/color/image_raw
/camera/depth/image_rect_raw
/tf
/tf_static
```

记录指定 Topic：

```bash
ros2 bag record -o demo_001 \
  /joint_states \
  /camera/color/image_raw \
  /camera/depth/image_rect_raw \
  /tf \
  /tf_static
```

按 `Ctrl+C` 结束，然后查看或回放：

```bash
ros2 bag info demo_001
ros2 bag play demo_001
```

**安全提示**：回放 rosbag 可能重新发布原有控制命令。带有真实机械臂的系统应隔离控制 Topic 或断开执行通道，避免无意运动。

<a id="course-section-35"></a>

### 6.4 对机器人学习有用的数据字段

模仿学习常用轨迹可以写作：

`D = { (o_t, a_t) }_(t=0,...,T)`

建议记录：

| 类别 | 示例 |
|---|---|
| 图像 | 头部 RGB、腕部 RGB、头部 Depth |
| 机器人状态 | 机械臂与手指关节位置、速度、末端位姿 |
| 动作 | 实际下发的命令、目标关节位置或速度 |
| 时间 | 每个数据源的原始时间戳 |
| 标定 | 相机内参、外参、TF、机器人 URDF 版本 |
| Episode 信息 | 任务、开始/结束时间、成功与否 |
| 可选 | 触觉、力矩、电流、物体 Pose |

必须区分：

- **Commanded Action**：遥操作或策略发出的目标动作。
- **Executed State**：机器人真实传感器测量到的执行状态。

二者可能因延迟、控制器限制、饱和或碰撞而不同，不能直接混同。

<a id="course-section-36"></a>

### 6.5 多频率同步与数据集转换

常见管线：

```text
ROS Bag
   ▼
Read Topics
   ▼
Timestamp Alignment
   ▼
Episode Segmentation
   ▼
Observation / Action Pairing
   ▼
HDF5 / Parquet / LeRobot-compatible Dataset
   ▼
Policy Training
```

示意频率（并非 Tron2 / Revo3 的设备规格）：

| 模态 | 示例频率 |
|---|---|
| RGB / Depth 相机 | 30 Hz |
| 机器人关节状态 | 100 Hz |
| 触觉 | 100 Hz |
| Policy Action | 50 Hz |

需要选择参考时间轴，对状态插值、对图像选择对应帧或按具体策略进行时序堆叠。**不能只根据消息到达顺序进行同步；应优先使用有意义的消息时间戳，并考虑硬件时钟偏差。**

---

<a id="course-section-37"></a>

## 七、URDF、TF2、RViz 和 ros2_control

<a id="course-section-38"></a>

### 7.1 URDF：机器人结构描述

URDF（Unified Robot Description Format）描述机器人连杆、关节与几何、惯性等信息。例如：

```text
Tron2
 ├── Shoulder Joint
 ├── Elbow Joint
 ├── Wrist Joint
 └── Revo3 Hand
      ├── Finger / Joint 1
      ├── Finger / Joint 2
      └── ...
```

典型 XML 元素：

```xml
<link>       <!-- 刚体连杆 -->
<joint>      <!-- 关节和连接关系 -->
<visual>     <!-- 显示几何 -->
<collision>  <!-- 碰撞几何 -->
<inertial>   <!-- 质量和惯量 -->
```

URDF 可为 ROS 可视化、运动学和仿真模型构建提供基础，但导入仿真器时可能还需要格式转换和补充参数。

<a id="course-section-39"></a>

### 7.2 URDF 与 TF2 的关系

**URDF 是结构描述；TF2 是带时间关系的坐标变换系统。**

```text
          URDF
            │
            ▼
robot_state_publisher ← /joint_states
            │
            ▼
         TF Tree
            │
            ▼
           RViz
```

在 Tron2 项目中，这一流程适合检查机械臂末端、Revo3 手基座、头部相机及腕部相机的位置关系。URDF 外参错误不会被 TF2 自动修复。

<a id="course-section-40"></a>

### 7.3 RViz 2：可视化工具

启动：

```bash
rviz2
```

RViz 能显示 Robot Model、TF、相机图像、点云、关节状态、姿态标记和运动轨迹。对于 FoundationPose 结果，可以同时检查物体 Pose 与机器人基座的相对关系。

<a id="course-section-41"></a>

### 7.4 ros2_control：连接高层命令与硬件

```text
High-Level Policy
        ▼
Controller (ros2_control)
        ▼
Hardware Interface
        ▼
Vendor SDK / Driver
        ▼
Motor / Robot
```

Hardware Interface 将读写硬件能力映射为控制框架接口；Controller 则执行轨迹、位置、速度或其他控制逻辑。

| 常见控制器或概念 | 用途 |
|---|---|
| Joint Trajectory Controller | 执行关节轨迹 |
| Forward Command Controller | 将指令传给指定关节接口 |
| Joint Position / Velocity 命令接口 | 发送位置或速度目标，取决于配置 |
| Diff Drive Controller | 差速移动底盘控制 |

安装：

```bash
sudo apt install \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers
```

启动兼容的 `controller_manager` 之后可查看：

```bash
ros2 control list_controllers
```

**重要：安装 ros2_control 本身不代表已经具备 Tron2 / Revo3 的硬件适配。** 必须找到兼容 Hardware Interface，或基于厂商 SDK 编写适配器。也可以在不使用 ros2_control 的情况下封装独立 ROS 2 驱动节点。

---

<a id="course-section-42"></a>

## 八、ROS 如何连接 Isaac Lab / MuJoCo 强化学习？

<a id="course-section-43"></a>

### 8.1 仿真训练通常不需要 ROS

Isaac Lab、MuJoCo 中的 PPO/SAC 训练往往直接调用环境 API：

```python
observations = env.reset()
for step in range(num_steps):
    actions = policy(observations)
    observations, rewards, dones, info = env.step(actions)
```

上面是概念示意，不对应所有 Gymnasium / Isaac Lab 版本的真实返回签名。大量并行环境需要高吞吐量，因此通常不把每个仿真步都转换为 ROS Topic。

<a id="course-section-44"></a>

### 8.2 真机部署需要连接各个模块

```text
Isaac Lab / MuJoCo
        ▼
  PPO / SAC Training
        ▼
   Exported Policy
        ▼
  ROS 2 Policy Node
        ▼
  Safety / Controller
        ▼
    Real Robot
```

典型真机闭环：

```text
Camera ────────┐
Joint States ──┼──→ Observation Builder → RL Policy
Object Pose ───┘                             │
                                            ▼
                                       Safety Filter
                                            ▼
                                     Robot Controller
                                            ▼
                                        Real Robot
```

<a id="course-section-45"></a>

### 8.3 Sim2Real 最关键的是接口语义一致

| 项目 | 训练端 | 真机端应做的事 |
|---|---|---|
| Joint Order | 仿真模型的关节顺序 | 显式映射为实际驱动顺序 |
| Joint Units | 常为 rad | 核对 SDK 是否使用 rad、degree 等 |
| Observation | 归一化状态 | 保持相同归一化与拼接顺序 |
| Action | 目标位置、增量或速度等 | 正确映射到控制模式 |
| Frequency | 仿真 timestep / decimation | 明确 Policy 与 Controller 周期 |
| Coordinate Frames | Sim World、Base 等 | 明确真实基座与世界系关系 |

ROS 能统一通信接口，却不会自动解决关节顺序、相机外参、归一化、动力学误差或延迟问题。

<a id="course-section-46"></a>

### 8.4 控制频率与延迟

需要分别考虑：

- `f_camera`：相机采样频率。
- `f_policy`：神经网络策略推理与动作更新频率。
- `f_control`：底层控制器频率。

这些频率可以不同。例如 30 Hz 的图像、50 Hz 的策略与更高频率的底层控制；这只是示意而非设备约束。

Python 的 ROS 2 节点不是天然硬实时系统。为减少延迟与提高稳定性，可分离 GPU 推理与伺服控制，使用 C++ / 实时线程完成关键控制，防止控制线程阻塞，并设置命令超时、限位和急停。

---

<a id="course-section-47"></a>

## 九、Tron2 + Revo3 灵巧操作项目架构

任务：**机器人从桌面拿起螺丝刀，移动到指定位置，完成旋转。** 系统包含 Tron2、BrainCo Revo3、头部 RGB-D、腕部 RGB、FoundationPose 和 PPO/SAC 策略。

<a id="course-section-48"></a>

### 9.1 推荐节点与信息流

```text
                    ROS 2 System
                          │
       ┌──────────────────┼───────────────────┐
       ▼                  ▼                   ▼
   Head Camera        Wrist Camera         Robot Driver
       │                  │                   │
       ▼                  ▼                   ▼
    RGB / Depth          RGB              Joint States
       └─────────┬────────┘                   │
                 ▼                            │
          Pose Estimator                      │
          (FoundationPose)                    │
                 │                            │
                 ▼                            │
             Object Pose                      │
                 └──────────────┬─────────────┘
                                ▼
                       Observation Builder
                                ▼
                            RL Policy
                                ▼
                         Safety Filter
                                ▼
                       Robot Driver / SDK
                                ▼
                         Tron2 + Revo3
```

<a id="course-section-49"></a>

### 9.2 推荐 Workspace 组织

```text
robot_learning_ws/
└── src/
    ├── tron2_driver/
    ├── revo3_driver/
    ├── camera_driver/
    ├── robot_description/
    ├── pose_estimation/
    ├── observation_builder/
    ├── rl_policy/
    ├── safety_controller/
    ├── teleoperation/
    └── data_recorder/
```

这是建议的工程目录，不代表厂商提供同名官方 Package。如果厂商只提供 Python/C++ SDK，可以编写 ROS Wrapper 转换为统一的消息接口。

<a id="course-section-50"></a>

### 9.3 机械臂与灵巧手的动作分发

联合策略可以输出：

`a_t = [ a_t^arm, a_t^hand ]`

其中机械臂动作维度为 `n_a`，灵巧手动作维度为 `n_h`。分发方案：

```text
          RL Policy
              ▼
           Action
          ┌───┴───┐
          ▼       ▼
      Arm Cmd   Hand Cmd
          ▼       ▼
     Tron2 SDK  Revo3 SDK
```

可用两个独立命令 Topic，也可以定义统一 Robot Action 消息由一个控制节点分发。无论哪种方案，都需要明确两个设备的控制时钟、关节顺序、目标模式和异常处置。

<a id="course-section-51"></a>

### 9.4 FoundationPose 的 ROS 接入

```text
Head RGB-D / Wrist RGB
          ▼
     FoundationPose
          ▼
      /object_pose
          ▼
     TF Transform
          ▼
Object Pose in Robot Base
          ▼
      RL Policy
```

位姿估计模块可以独立于控制器开发，只需保证输出的 PoseStamped 时间戳与 frame_id 准确，必要时考虑遮挡、估计丢失后的重置与时间滤波。

---

<a id="course-section-52"></a>

## 十、机器人学习中的 ROS 常见问题

| 现象 | 可能原因 | 排查方向 |
|---|---|---|
| `ros2 topic list` 看不到设备 | 驱动未启动、网络发现或域配置问题 | 检查节点、DDS、`ROS_DOMAIN_ID` |
| Topic 存在但收不到数据 | QoS 不兼容、订阅类型不一致 | `ros2 topic info -v` |
| 相机和关节数据不同步 | 不同频率、时钟不一致 | 时间戳、硬件同步、缓存和插值 |
| 物体 Pose 偏移 | 外参、TF 链或参考系定义有误 | 检查标定与 optical frame |
| RL 真机动作异常 | 关节顺序、归一化、动作模式不一致 | 对比训练与部署 Observation/Action |
| 端到端延迟大 | 相机、推理、通信、SDK 或控制器延迟 | 分段计时、测量队列积压 |
| ROS 与 Conda 冲突 | 系统 Python 与 Conda 的 ABI/依赖不匹配 | 分离进程或使用兼容环境 |
| rosbag 文件过大 | 多路 RGB-D 带宽较高 | 压缩、MCAP/存储设置、拆分 |
| RViz 模型不对 | URDF、JointState、TF 不一致 | 逐级核对 Robot Model / TF |

<a id="course-section-53"></a>

### 10.1 ROS 2 与 Conda 的关系

机器人学习经常在 Conda 环境中使用 PyTorch、CUDA、Isaac Lab 和 RL 框架；而通过 apt 安装的 `rclpy` 通常面向 Ubuntu 系统 Python。直接在任意 Conda 环境里 `import rclpy` 可能遇到 ABI 或动态库问题。

一种稳妥结构是：

```text
 ROS 2 Driver Process
          │
       Observation
          ▼
PyTorch Inference Process
    (Conda / CUDA)
          │
         Action
          ▼
 ROS 2 Control Process
```

可以通过 ROS 2 Topic、独立通信接口或共享内存传递数据，而非强行把全部依赖安装在同一个 Python 环境。

<a id="course-section-54"></a>

### 10.2 与真实机器人相关的安全边界

学习策略的输出不是可直接信任的底层电机命令。真实系统至少需要：关节限位、速度与加速度约束、命令超时、停止与急停、通信失联回退，以及在初始调试时从低速、无负载、无接触场景开始。ROS 本身不会替代这些安全机制。

---

<a id="course-section-55"></a>

## 十一、推荐学习路线

| 阶段 | 重点 | 可检验的完成标准 |
|---|---|---|
| 1 | Node、Topic、Message | 能查看并解释机器人的 Topic |
| 2 | `rclpy`、Publisher、Subscriber | 能运行自己的 Python 策略节点 |
| 3 | rosbag2 | 能完整记录并回放一个 Episode |
| 4 | URDF、TF2、RViz | 能核对机器人与相机坐标系 |
| 5 | 机器人 SDK、ros2_control | 能安全读取状态并下发测试目标 |
| 6 | PyTorch + ROS | 能在真机运行 PPO/SAC 推理 |
| 7 | QoS、同步与延迟分析 | 能量化传感器到动作的端到端延迟 |

对于已经有机器人学习、控制和 Python 基础的研究者，不必从 ROS 的每个组件入手，应围绕 **数据采集 → 策略部署 → 可复现实验** 逐步学习。

---

<a id="course-section-56"></a>

## 十二、总结与实践建议

ROS 2 在机器人学习中的价值可以归为三类：

| 方向 | ROS 负责的工作 | 典型应用 |
|---|---|---|
| **数据采集** | 采集与记录相机、机器人状态和遥操作动作 | BC、Diffusion Policy、VLA |
| **策略部署** | 连接 Observation、Policy、Controller | PPO、SAC、Sim2Real |
| **系统集成** | TF、硬件通信、可视化、诊断 | 灵巧操作、视觉闭环、真机实验 |

对于 **Tron2 + Revo3**，最优先掌握的五项是：

1. **Topic + rclpy**：连接关节状态、相机与 RL 策略。
2. **rosbag2**：采集遥操作数据，构建离线训练数据集。
3. **TF2 + URDF**：管理头部相机、腕部相机、末端和物体坐标系。
4. **Robot SDK / Hardware Interface**：可靠控制 Tron2 与 Revo3。
5. **Sim2Real Deployment**：将 Isaac Lab / MuJoCo 中的策略转换为真机闭环控制。

**最终目标不是“在训练中使用更多 ROS”，而是建立可观测、可复现、便于维护的“传感器 → 策略 → 控制器 → 机器人”系统。**

---

<a id="course-section-57"></a>

## 十三、官方文档与参考资料

1. [ROS 2 Jazzy 官方教程](https://docs.ros.org/en/jazzy/Tutorials.html)
2. [ROS 2 Jazzy Ubuntu deb 安装说明](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)
3. [ROS 2 Python Publisher / Subscriber 教程](https://docs.ros.org/en/jazzy/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Publisher-And-Subscriber.html)
4. [ROS 2 TF2 教程](https://docs.ros.org/en/jazzy/Tutorials/Intermediate/Tf2/Tf2-Main.html)
5. [ros2_control 官方文档](https://control.ros.org/jazzy/)
6. [MoveIt 2 官方文档](https://moveit.picknik.ai/)
7. [rosbag2 GitHub](https://github.com/ros2/rosbag2)
8. [ROS 2 发行版本信息](https://github.com/ros2/ros2_documentation/blob/rolling/source/Releases.rst)
9. [ROS 2 Lyrical 官方发行记录](https://github.com/ros2/ros2/releases)

**文档说明**：本教程为学习与工程设计材料，代码中的虚拟动作 Topic 仅供无硬件示例测试；真实机器人部署必须结合厂商接口文档和设备安全要求检查。
