from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from social_nav3d.env.sim import SocialNavSim
from social_nav3d.planners.sampling_mpc import SamplingMPC
from social_nav3d.utils.config import load_config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='social_nav3d/configs/default.yaml')
    ap.add_argument('--gui', action='store_true')
    ap.add_argument('--record', action='store_true')
    ap.add_argument('--steps', type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.gui:
        cfg.setdefault('sim', {})
        cfg['sim']['gui'] = True
    if args.record:
        cfg.setdefault('sim', {})
        cfg['sim']['record_video'] = True

    sim = SocialNavSim(cfg)
    sim.reset()

    planner = SamplingMPC(cfg)

    traj = []
    for k in range(args.steps or sim.max_steps):
        state = sim.get_state()
        ped_states = sim.get_pedestrians_state()
        peds = [(d['xy'], d['yaw'], d['vel'], d['ps']) for d in ped_states]
        lidar_min = sim.lidar_min_distance()
        v, w = planner.plan(state, sim.goal_xy, sim.dt, lidar_min, peds)
        sim.step(v, w)
        traj.append([state.x, state.y])
        if sim.reached_goal():
            break

    out_dir = Path(cfg['sim'].get('out_dir', 'runs'))
    out_dir.mkdir(parents=True, exist_ok=True)

    traj = np.asarray(traj)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    sim.plot_topdown(ax)
    ax.plot(traj[:, 0], traj[:, 1])
    ax.scatter([traj[0, 0]], [traj[0, 1]], marker='o')
    ax.scatter([sim.goal_xy[0]], [sim.goal_xy[1]], marker='*')
    ax.set_title('SocialNav3D: trajectory')
    fig.tight_layout()
    fig.savefig(out_dir / 'trajectory.png', dpi=160)
    print(f"Saved: {out_dir / 'trajectory.png'}")

    if cfg['sim'].get('record_video', False):
        print(f"Saved: {out_dir / 'run.mp4'}")

    sim.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
