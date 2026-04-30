# SSM-MPC-Experiments

软体机器人轨迹跟踪实验代码，包含基于 SSM 的 MPC 闭环控制主程序、PCC 逆运动学基线程序、MATLAB 原始工程参考文件，以及实验数据绘图脚本。

本仓库当前以 Python 版本为主。`MATLAB/Snake_experiment_Eight` 保留了原 MATLAB/Simulink 工程，用于对照和追溯；已迁移的 Python 代码见 `Main_demo_SSM.py`、`controller/`、`dynamics/`、`measure/`、`utils/` 等目录。

## 功能概览

- `Main_demo_SSM.py`：SSM-MPC 实验主入口，读取 SSM ROM 数据，调用 LuMo SDK 和动捕数据，计算 6 路电机位移并通过 ZeroMQ 发布。
- `Main_demo_pcc.py`：PCC 逆运动学轨迹跟踪基线程序，读取 `traj_data/` 中的参考轨迹，输出 6 路绳驱电机命令。
- `controller/`：SSM-MPC 控制器、NMPC/QP 求解逻辑、绳长约束计算。
- `dynamics/`：PCC 逆运动学和 SSM 降阶动力学相关函数。
- `measure/`：LuMo 动捕 SDK 数据读取、基坐标系初始化、末端/中点位移计算。
- `zmq_message/`：电机控制消息封装和 ZeroMQ 发布器。
- `PlotFig/`：实验结果保存目录和绘图脚本。
- `MATLAB/`：原 MATLAB 工程参考文件。

## 运行环境

建议使用 Windows + Python 3.11。仓库中的 LuMo SDK Python 扩展为 `bin/LuMoSDKPy311.pyd`，因此主实验程序默认面向 Python 3.11。

Python 依赖：

```powershell
python -m pip install numpy scipy matplotlib pyzmq casadi
```

说明：

- `numpy`、`scipy`：矩阵计算、`.mat` 数据读取、优化和矩阵指数。
- `matplotlib`：绘图脚本使用。
- `pyzmq`：向电机控制端发布控制命令。
- `casadi`：完整 NMPC 求解使用。当前 `Main_demo_SSM.py` 默认 `horizon = 1` 且 `USE_FAST_SINGLE_STEP_QP = True`，会优先走单步快速 QP 路径；如果改成多步预测或关闭快速路径，需要安装并正确配置 CasADi。

## 硬件与外部程序要求

运行闭环实验前，需要确认以下外部系统已准备好：

1. LuMo SDK 动捕服务可连接，默认地址为 `127.0.0.1`。
2. `bin/` 目录中的 LuMo SDK 动态库和 `.pyd` 文件存在。
3. 电机控制端正在订阅 ZeroMQ topic `/controller/setActuatorDate`。
4. 电机控制 ZeroMQ 地址与代码一致，默认 `tcp://localhost:5555`。
5. 动捕刚体和 marker ID 与实验装置一致。

关键 ID 默认值：

- PCC 程序：基座刚体 `0`，中点刚体 `1`，末端 marker `100028`。
- SSM 程序：基座刚体 `0`，中点刚体 `1`，末端 marker `100009`。

如实验现场 marker 编号不同，请修改 `Main_demo_SSM.py` 中的 `END_MARKER_ID`，或调用 `main(end_marker_id=...)`。

## 数据文件

SSM-MPC 使用的数据位于：

```text
dynamics/SSM/ROM8_Expe/
```

主要包括：

- `auData.mat`
- `W_0.mat`
- `Node_List.mat`
- `zd_proj.mat`
- `t_list.mat`
- `outdof_endeff.mat`
- `outdof_half.mat`

PCC 基线程序使用的数据位于：

```text
traj_data/
```

主要包括：

- `zd_proj.mat`
- `t_list.mat`

## 运行 SSM-MPC 实验

在仓库根目录运行：

```powershell
python Main_demo_SSM.py
```

程序流程：

1. 读取 `dynamics/SSM/ROM8_Expe/` 下的 SSM ROM 数据。
2. 构造 SSM 输出映射、绳长映射和反馈映射。
3. 连接 LuMo SDK 并初始化动捕参考坐标系。
4. 在 `tcp://localhost:5555` 创建 ZeroMQ publisher。
5. 按 `CONTROL_PERIOD_SECONDS = 0.075` 秒周期执行控制。
6. 每步采集末端和中点位移，计算 MPC 控制量，发送 6 路电机命令。
7. 保存实验数据和日志。

输出文件：

```text
PlotFig/y_exprm_ssm.npy
PlotFig/ut_save_ssm.npy
PlotFig/t_ssm.npy
PlotFig/ssm_python_results.mat
PlotFig/logs/SSM_log_*.txt
```

常用参数在 `Main_demo_SSM.py` 顶部：

- `CONTROL_PERIOD_SECONDS`：控制周期。
- `horizon`：MPC 预测步长。
- `RUN_STEPS`：运行步数，默认 `np.inf` 表示按参考轨迹长度运行。
- `FEEDBACK_ENABLED`：是否启用测量反馈修正。
- `MPC_SOLVER_TYPE`：可选 `"ipopt"` 或 `"qp"`。
- `QP_BACKEND`：QP 后端，默认 `"qpoases"`。
- `USE_FAST_SINGLE_STEP_QP`：当 `horizon = 1` 时启用快速单步 QP。
- `SSM_FACTOR`、`MOTOR_FACTOR`：SSM 映射和电机输出缩放系数。

## 运行 PCC 基线实验

在仓库根目录运行：

```powershell
python Main_demo_pcc.py
```

程序会读取 `traj_data/zd_proj.mat` 和 `traj_data/t_list.mat`，根据 PCC 逆运动学计算电机位移，通过 ZeroMQ 发布控制命令，并保存：

```text
PlotFig/y_expm.npy
PlotFig/t_expm.npy
PlotFig/q_save.npy
PlotFig/u_save.npy
```

## 绘图

SSM-MPC 结果：

```powershell
python PlotFig/plot_ssm_results.py
```

PCC 基线结果：

```powershell
python PlotFig/plot_pcc_results.py
```

绘图脚本默认读取 `PlotFig/` 下最近由主程序保存的 `.npy` 文件。

## MATLAB 到 Python 的迁移说明

项目相关 MATLAB 逻辑已经迁移到 Python 目录：

- MATLAB SSM 控制逻辑：`controller/`
- MATLAB SSM 模型算法和数据接口：`dynamics/SSM/`
- MATLAB ROM 数据：`dynamics/SSM/ROM8_Expe/`
- MATLAB 主入口对应 Python 文件：`Main_demo_SSM.py`
- MATLAB 绘图对应 Python 文件：`PlotFig/plot_ssm_results.py`

`MATLAB/Snake_experiment_Eight/casadi/` 是 MATLAB 版第三方 CasADi 发行目录，没有逐文件迁移。Python 版本使用 Python 包 `casadi`。

## 注意事项

- 运行主实验会向电机控制端发送真实控制命令，调试前请确认急停、限位和控制端订阅状态。
- `SsmMpcController` 内部包含电机输出保护：若缩放后的任一电机位移绝对值超过 `100 mm`，本步输出会置零。
- `Main_demo_SSM.py` 默认关闭测量反馈修正；如需启用，将 `FEEDBACK_ENABLED` 改为 `True`。
- 若只想检查绘图或数据读取，可以先运行 `PlotFig/` 下脚本；这些脚本不连接硬件。
- 如果 Python 无法导入 `LuMoSDKPy311`，请确认正在使用 Python 3.11，并且 `bin/` 中的 DLL 文件没有缺失。
