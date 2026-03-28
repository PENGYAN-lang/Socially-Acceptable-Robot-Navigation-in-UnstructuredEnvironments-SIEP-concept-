# SocialNav3D – SIEP-based Socially-Acceptable Robot Navigation

A runnable 3D physics demo (PyBullet) for socially-aware navigation in
unstructured environments.  The project now implements the **Stimuli-Induced
Equilibrium Point (SIEP)** concept described in the project specification,
replacing the earlier pure MPC reward/penalty approach with an analytically
grounded virtual-force framework.

---

## Project Overview

**Project Code**: 2520-00014  
**Title**: Socially Acceptable Robot Navigation in Unstructured Environments

Autonomous robots operating in spaces such as shopping malls or transport hubs
must navigate safely *and* in a socially acceptable manner — respecting
personal space, anticipating motion, and matching crowd flow.  This demo adopts
the SIEP concept to model the robot's navigation behaviour from first
principles: each perceived stimulus (goal, obstacle, pedestrian) induces a
virtual force, and the superposition of all forces defines an *equilibrium
point* that the robot tracks.

---

## SIEP Framework

### Core Idea

In the SIEP framework the robot's motion dynamics are characterised as the
result of virtual forces induced by stimuli perceived from the surrounding
environment.  The net force defines a desired velocity vector (the equilibrium
point); a proportional heading controller converts this into differential-drive
commands.

```
F_total = F_goal + F_obstacle + F_personal_space + F_velocity_alignment
```

### Stimulus Channels

| Stimulus | Formula | Innovation |
|---|---|---|
| **Goal attraction** | Tanh-saturated pull: `k·tanh(d/σ)·(goal−pos)/d` | Smooth deceleration near goal, bounded cruise speed |
| **Obstacle repulsion** | Exponential push per LiDAR ray: `k·exp(−d/λ)·(−ray_dir)` | Directional, proportional to proximity |
| **Personal-space repulsion** | Predictive anisotropic Gaussian integral over future pedestrian positions (discounted) | **Predictive SIEP**: avoids violations *before* they occur |
| **Velocity alignment** | Weighted velocity-difference toward nearby same-direction pedestrians | **Crowd-flow following**: new stimulus absent in plain MPC |

### Innovations Over Sampling MPC

1. **Predictive personal-space forces** — the social repulsion is integrated
   over a configurable look-ahead horizon (default 1.5 s), so the robot steers
   away from pedestrians before entering their personal space.  Each predicted
   step is time-discounted so that the immediate threat matters most.

2. **Velocity-alignment stimulus** — a new force channel that gently nudges
   the robot to match the speed and direction of nearby pedestrians walking the
   same way.  This captures crowd-flow awareness not present in pure
   repulsion/penalty models.

3. **Tanh-saturated goal attraction** — the goal force saturates at the
   desired cruise speed rather than growing without bound, preventing
   speed-overshoot in long corridors and providing smooth goal approach.

4. **Deterministic, interpretable planning** — unlike sampling MPC, forces are
   computed analytically, making the robot's behaviour directly explainable in
   terms of contributing stimuli.

---

## Repository Structure

```
social_nav3d/
├── env/
│   └── sim.py                  # PyBullet simulator + get_lidar_directional_2d()
├── planners/
│   ├── siep_planner.py         # ★ SIEP equilibrium-point planner (new)
│   └── sampling_mpc.py         # Original sampling MPC (retained)
├── utils/
│   ├── siep_forces.py          # ★ SIEP virtual-force kernels (new)
│   ├── social.py               # Anisotropic Gaussian personal-space model
│   └── geometry.py             # Pose2, unicycle integration
└── configs/
    ├── siep_demo.yaml           # ★ SIEP planner configuration (new)
    └── default.yaml             # MPC configuration
```

---

## Setup

### Requirements
- **Python**: 3.10 (3.9–3.11 compatible)
- **PyBullet**, NumPy, Matplotlib, SciPy, PyYAML (see `requirements.txt`)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Conda alternative:
```bash
conda create -n socialnav3d python=3.10 -y
conda activate socialnav3d
pip install -r requirements.txt
```

### Optional: PyTorch (for future learned prediction)
```bash
# Example for CUDA 12.1 — adjust for your driver
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
```

---

## Run

### SIEP planner (recommended)
```bash
# Headless
python run_demo.py --config social_nav3d/configs/siep_demo.yaml --planner siep

# GUI
python run_demo.py --config social_nav3d/configs/siep_demo.yaml --planner siep --gui
```

### Original sampling MPC (for comparison)
```bash
python run_demo.py --config social_nav3d/configs/default.yaml --planner mpc
```

Output trajectory saved to `runs/trajectory.png`.

---

## Configuration

Key SIEP planner parameters in `social_nav3d/configs/siep_demo.yaml`:

| Parameter | Default | Description |
|---|---|---|
| `k_goal` | 1.0 | Cruise speed / goal force magnitude [m/s] |
| `sigma_goal` | 3.0 | Tanh saturation distance [m] |
| `k_obs` | 2.5 | Obstacle repulsion scale |
| `obs_influence` | 2.5 | Max obstacle influence radius [m] |
| `k_ps` | 2.0 | Personal-space repulsion scale |
| `ps_horizon` | 1.5 | Predictive look-ahead [s] |
| `k_align` | 0.3 | Velocity-alignment scale |
| `align_radius` | 3.0 | Alignment effective radius [m] |
| `align_cone_deg` | 60.0 | Same-direction cone half-angle [deg] |
| `k_yaw` | 2.5 | Heading proportional gain |
| `heading_slowdown` | 0.5 | Speed reduction factor during turns |

To increase scenario complexity, edit the YAML:
- `pedestrians.count` — more pedestrians
- `world.obstacles` — add boxes/ramps
- `lidar.n_az` — denser LiDAR
- `ps_horizon` — longer prediction horizon

---

## 在启智 (OpenI) 平台上运行

> **推荐方式：使用 Jupyter Notebook，全程无需手动敲命令。**

### 方式 A：直接打开 Notebook（最简单）

1. 登录 [启智 (OpenI)](https://openi.pcl.ac.cn/)，进入你的项目。
2. 在顶部导航选择 **「云脑」→「调试任务」**（或 **JupyterLab**）。
3. 启动一个 CPU 或 GPU 调试实例（CPU 实例已够用于本仿真）。
4. 调试实例启动后，打开终端，执行：

   ```bash
   git clone https://github.com/PENGYAN-lang/Socially-Acceptable-Robot-Navigation-in-UnstructuredEnvironments-SIEP-concept-.git SocialNav3D
   ```

5. 在 JupyterLab 左侧文件浏览器里，依次进入 `SocialNav3D/`，双击打开 **`SIEP_demo.ipynb`**。
6. 顶部菜单选 **`Run → Run All Cells`**，等待几分钟，轨迹图会直接显示在 Notebook 里。

### 方式 B：命令行运行

在调试实例的终端里依次执行：

```bash
# 1. 克隆仓库
git clone https://github.com/PENGYAN-lang/Socially-Acceptable-Robot-Navigation-in-UnstructuredEnvironments-SIEP-concept-.git SocialNav3D
cd SocialNav3D

# 2. 安装依赖（约 1-2 分钟）
pip install -r requirements.txt

# 3. 运行 SIEP 仿真
python run_demo.py --config social_nav3d/configs/siep_demo.yaml --planner siep

# 4. 对比原始 MPC（可选）
python run_demo.py --config social_nav3d/configs/default.yaml --planner mpc
```

运行结果（轨迹图）保存在 `runs/trajectory.png`，可在 JupyterLab 文件浏览器里直接预览。

### 常见问题

| 问题 | 解决办法 |
|---|---|
| `pybullet` 安装失败 | 先 `pip install --upgrade pip` 再重试 |
| `ModuleNotFoundError: social_nav3d` | 确保在项目根目录（有 `run_demo.py` 的目录）运行 |
| 运行时报 `display` 错误 | 仿真不需要 GUI，确认命令里没有 `--gui` 参数 |
| 调试实例资源不够 | 在启智创建任务时选择 CPU ≥ 4 核，内存 ≥ 8 GB |
