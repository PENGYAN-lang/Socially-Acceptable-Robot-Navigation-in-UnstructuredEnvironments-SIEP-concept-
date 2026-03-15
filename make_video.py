import os
import math
import time
import argparse
import numpy as np
import imageio
import yaml
import pybullet as p

from social_nav3d.env.sim import SocialNavSim
import torch
from social_nav3d.planners.gpu_social_mpc import GPUSocialMPC



def safe_render(client_id, target_pos, width=1280, height=720):
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target_pos,
        distance=9,
        yaw=45,
        pitch=-35,
        roll=0,
        upAxisIndex=2
    )
    proj = p.computeProjectionMatrixFOV(60, width / height, 0.1, 120.0)
    w, h, rgba, _, _ = p.getCameraImage(
        width, height, view, proj,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=client_id
    )
    rgb = np.reshape(rgba, (h, w, 4))[:, :, :3]
    return rgb


def pose2_to_xytheta(pose2):
    """
    Robustly extract x,y,theta from a Pose2-like object.
    """
    # common attribute names
    for ax, ay, ath in (("x", "y", "theta"), ("x", "y", "yaw")):
        if hasattr(pose2, ax) and hasattr(pose2, ay) and hasattr(pose2, ath):
            return float(getattr(pose2, ax)), float(getattr(pose2, ay)), float(getattr(pose2, ath))

    # sometimes pose2.p.x pose2.p.y and pose2.theta
    if hasattr(pose2, "p") and hasattr(pose2.p, "x") and hasattr(pose2.p, "y"):
        th = float(getattr(pose2, "theta", 0.0))
        return float(pose2.p.x), float(pose2.p.y), th

    # fallback: try tuple/list conversion
    try:
        arr = list(pose2)
        if len(arr) >= 3:
            return float(arr[0]), float(arr[1]), float(arr[2])
    except Exception:
        pass

    # last resort
    return 10.0, 10.0, 0.0


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def get_goal_from_cfg(cfg):
    """
    Try to find goal in config dict. If not found, fallback.
    """
    # common layouts
    candidates = [
        ("goal",),
        ("task", "goal"),
        ("sim", "goal"),
        ("world", "goal"),
        ("env", "goal"),
        ("scenario", "goal"),
        ("robot", "goal"),
    ]
    for path in candidates:
        d = cfg
        ok = True
        for k in path:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                ok = False
                break
        if ok:
            # allow [x,y] or {x:..,y:..}
            if isinstance(d, (list, tuple)) and len(d) >= 2:
                return float(d[0]), float(d[1])
            if isinstance(d, dict) and "x" in d and "y" in d:
                return float(d["x"]), float(d["y"])
    return 10.0, 10.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", default="runs")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--kv", type=float, default=0.8, help="linear speed gain")
    parser.add_argument("--kw", type=float, default=1.8, help="angular speed gain")
    parser.add_argument("--vmax", type=float, default=1.2)
    parser.add_argument("--wmax", type=float, default=2.0)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    goal_x, goal_y = get_goal_from_cfg(cfg)
    
    sim = SocialNavSim(cfg)
    sim.reset()
    world = cfg["world"]  # 取静态障碍和地图信息

    planner = GPUSocialMPC(device="cuda",
                            horizon=25,
                            samples=8192,
                            dt=0.1,
                            v_limits=(0.0, args.vmax),
                            w_limits=(-args.wmax, args.wmax),
                            robot_radius=0.35,
                            w_goal=2.0,
                            w_smooth=0.2,
                            w_ps=6.0,
                            w_coll=80.0,
                            w_obs=200.0,
                            w_bound=200.0)
    assert torch.cuda.is_available()
    torch.cuda.synchronize()
    print("[gpu] planner ready")
    _ = torch.empty((1024,1024), device="cuda")
    torch.cuda.synchronize()
    print("[gpu] warmup done", flush=True)
        

    client_id = getattr(sim, "client", None)
    if not isinstance(client_id, int):
        raise RuntimeError(f"sim.client is not an int physicsClientId: {type(client_id)}")

    frames = []
    traj = []
    t0 = time.time()

    for step in range(args.steps):
        # --- use GPU Social MPC ---
        pose = sim.get_state()
        x, y, th = pose2_to_xytheta(pose)  # 或调用 pose2_to_xytheta()
        frames.append(safe_render(client_id, [x, y, 1.0]))
        traj.append((x, y, 0.0))

        peds = sim.get_pedestrians_state()
        t_gpu_start = time.time()
        v, w = planner.act(pose, peds, (goal_x, goal_y), world=world)
        if step % 20 == 0:
            torch.cuda.synchronize()
            gpu_ms = (time.time() - t_gpu_start) * 1000
            print(f"[gpu] step={step} act_ms={gpu_ms:.2f}", flush=True)
        sim.step(float(v), float(w))

        # 距离目标
        dist = math.hypot(goal_x - x, goal_y - y)
        if dist < 0.3:
            break


    mp4_path = os.path.join(args.outdir, "demo.mp4")
    imageio.mimsave(mp4_path, frames, fps=args.fps)
    print(f"[OK] saved video: {mp4_path}")

    # 2D + 3D plots
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        traj = np.array(traj, dtype=np.float32)

        # 2D top-down
        plt.figure()
        plt.plot(traj[:, 0], traj[:, 1])
        plt.scatter([traj[0,0]],[traj[0,1]], marker="o")
        plt.scatter([goal_x],[goal_y], marker="*")
        plt.xlabel("x"); plt.ylabel("y"); plt.title("Trajectory (top-down)")
        plt.savefig(os.path.join(args.outdir, "traj_xy.png"), dpi=200)

        # 3D (z=0)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2])
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title("Trajectory (3D view, z=0)")
        plt.savefig(os.path.join(args.outdir, "traj_3d.png"), dpi=200)

        print("[OK] saved plots: traj_xy.png, traj_3d.png")
    except Exception as e:
        print("[WARN] plot skipped:", str(e))


if __name__ == "__main__":
    main()