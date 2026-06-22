#!/usr/bin/env python3
"""
run_robot_conditional.py

1-D UNet

Conditional diffusion trainer for short-horizon robot trajectories.
Conditions: start (x, y, theta, v), goal (x, y), up to K_max dynamic obstacles (x,y,vx,vy).
Output: trajectory (x,y) over 16 samples (take each to be 0.1 s long so 1.6 s horizon)

TO START A COMPUTE NODE, DO IN TERMINAL: srun --mem=8G --time=1:00:00 --pty bash
THEN TO EXIT, CTRL+D
THIS WAY, YOU AREN'T IN LOGIN, CORRECTLY USE A DEDICATED NODE FOR COMPS
"""

import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import math
import wandb
from torchviz import make_dot

def log_trajectory_scene(
    gt,
    pred,
    start=None,
    goal=None,
    obs_pos=None,
    obs_vel=None,
    step=0,
    prefix="sample",
):
    """
    gt, pred: [T, 2]
    start, goal: [2]
    obs_pos: [N, 2]
    obs_vel: [N, 2]
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(5, 5))

    # GT and predicted trajectories
    ax.plot(gt[:, 0], gt[:, 1], "o-", label="GT", alpha=0.8)
    ax.plot(pred[:, 0], pred[:, 1], "x--", label="Pred", alpha=0.8)

    # Start / Goal
    if start is not None:
        if len(start) >= 2:
            ax.scatter(start[0], start[1], c="green", s=100, marker="o", label="Start")
        else:
            ax.scatter(0, 0, c="green", s=100, marker="o", label="Start")

    if goal is not None:
        ax.scatter(goal[0], goal[1], c="red", s=100, marker="*", label="Goal")

    # Obstacles
    if obs_pos is not None:
        ax.scatter(
            obs_pos[:, 0],
            obs_pos[:, 1],
            c="black",
            s=60,
            marker="s",
            label="Obstacle",
        )

        # Velocity arrows
        if obs_vel is not None:
            ax.quiver(
                obs_pos[:, 0],
                obs_pos[:, 1],
                obs_vel[:, 0],
                obs_vel[:, 1],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.003,
                color="black",
                alpha=0.8,
            )

    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"Trajectory @ step {step}")
    ax.grid(True)

    wandb.log(
        {f"{prefix}/scene": wandb.Image(fig)},
        step=step,
    )

    plt.close(fig)



# ------------------------
# Dataset - supports variable obstacles via padding and mask
# ------------------------
class RobotTrajectoryDataset(Dataset):
    def __init__(self, dataset_dir, horizon=16, k_max=10, norm_stats = None, files=None):
        """
        dataset_dir: directory with .npz or .npy files
        horizon: number of timesteps per training segment
        k_max: maximum number of obstacles to encode (pad/truncate)
        """
        if files is None:
            files = sorted(glob.glob(os.path.join(dataset_dir, "*.npz"))
                           + glob.glob(os.path.join(dataset_dir, "*.npy")))
        self.files = files
        if len(files) == 0:
            raise ValueError(f"No .npz/.npy files found in {dataset_dir}")
        self.horizon = horizon
        self.k_max = k_max
        self.norm_stats = norm_stats

    def __len__(self):
        return len(self.files)

    def _load_file(self, path):
        if path.endswith(".npz"):
            data = np.load(path, allow_pickle=True)
            traj = data["trajectory"]
            start = data["start_state"] if "start_state" in data else traj[0]
            goal = data["goal"] if "goal" in data else traj[-1]
            obstacles = data["obstacles"] if "obstacles" in data else np.zeros((0,4), dtype=np.float32)
        else:
            # single npy file assumed to be trajectory only
            traj = np.load(path)
            start = traj[0]
            goal = traj[-1]
            obstacles = np.zeros((0,4), dtype=np.float32)
        return traj.astype(np.float32), start.astype(np.float32), goal.astype(np.float32), obstacles.astype(np.float32)

    def __getitem__(self, idx):
        traj, start, goal, obstacles = self._load_file(self.files[idx])
        T_full = traj.shape[0]

        # choose a segment of length horizon
        if T_full >= self.horizon:
            # pick random segment
            i = np.random.randint(0, T_full - self.horizon + 1)
            seg = traj[i:i+self.horizon]
        else:
            # pad by repeating last state
            pad_len = self.horizon - T_full
            pad = np.repeat(traj[-1:,:], pad_len, axis=0)
            seg = np.concatenate([traj, pad], axis=0)

        # process obstacles: pad/truncate to k_max, and mask
        N = obstacles.shape[0]
        obs_trunc = np.zeros((self.k_max, 4), dtype=np.float32)
        obs_mask = np.zeros((self.k_max,), dtype=np.float32)

        if N > 0:
            n_to_copy = min(N, self.k_max)
            obs_trunc[:n_to_copy, :] = obstacles[:n_to_copy, :4]
            obs_mask[:n_to_copy] = 1.0

        # Apply normalization if stats provided
        if self.norm_stats is not None:
            start = (start - self.norm_stats["start_mean"]) / self.norm_stats["start_std"]
            goal  = (goal  - self.norm_stats["goal_mean"])  / self.norm_stats["goal_std"]
            seg  = (seg  - self.norm_stats["traj_mean"])  / self.norm_stats["traj_std"]
        
            valid = obs_mask.astype(bool)
            obs_trunc[valid] = (obs_trunc[valid] - self.norm_stats["obs_mean"]) / self.norm_stats["obs_std"]
            # obs_trunc = (obs_trunc - self.norm_stats["obs_mean"]) / self.norm_stats["obs_std"]

        sample = {
            "trajectory": torch.from_numpy(seg),         # (horizon, dim)
            "start_state": torch.from_numpy(start),           # (start_dim,)
            "goal": torch.from_numpy(goal),             # (goal_dim,)
            "obstacles": torch.from_numpy(obs_trunc),   # (k_max, 4)
            "obs_mask": torch.from_numpy(obs_mask)      # (k_max,)
        }
        return sample

def timestep_embedding(timesteps, dim):
    """
    Create sinusoidal timestep embeddings.
    timesteps: [B] integer timesteps
    dim: embedding dimension
    Returns: [B, dim]
    """
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0,1))
    return emb
# ------------------------
# 1D temporal convolution (Temporal UNet-style) with conditioning
# Should work better than the MLP first attempted
# ------------------------
class ConditionalTemporalNet(nn.Module):
    def __init__(self, traj_dim, cond_dim, hidden_dim=256, num_layers=5, kernel_size=3):
        """
        traj_dim: trajectory dimension per timestep (e.g., 2)
        cond_dim: conditioning vector dimension
        hidden_dim: number of channels for Conv1d
        num_layers: number of temporal conv layers
        kernel_size: size of temporal conv kernel
        """
        super().__init__()
        self.traj_dim = traj_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.time_embed_dim = hidden_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # input channels = traj_dim + cond_dim
        in_ch = traj_dim + cond_dim + hidden_dim

        layers = []
        for i in range(num_layers):
            layers.append(
                nn.Conv1d(
                    in_channels=in_ch if i==0 else hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size//2
                )
            )
            layers.append(nn.ReLU())
        # final layer to project back to traj_dim
        layers.append(nn.Conv1d(hidden_dim, traj_dim, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_noisy, t=None, cond=None):
        """
        x_noisy: [B, T, traj_dim]
        cond: [B, cond_dim] or [B, T, cond_dim]
        returns predicted noise: [B, T, traj_dim]
        """
        B, T, D = x_noisy.shape
        if cond is None:
            cond_exp = torch.zeros(B, T, 0, device=x_noisy.device)
        else:
            if cond.dim() == 2:
                cond_exp = cond.unsqueeze(1).expand(-1, T, -1)
            else:
                cond_exp = cond

        # ---- NEW: timestep embedding ----
        t_emb = timestep_embedding(t, self.time_embed_dim)  # [B, hidden_dim]
        t_emb = self.time_mlp(t_emb)                        # [B, hidden_dim]
        t_exp = t_emb.unsqueeze(1).expand(-1, T, -1)        # [B, T, hidden_dim]
        
        x = torch.cat([x_noisy, cond_exp, t_exp], dim=-1)  # [B, T, D+cond]
        x = x.permute(0, 2, 1)  # [B, C, T] for Conv1d
        out = self.net(x)
        out = out.permute(0, 2, 1)  # back to [B, T, D]
        return out

# ------------------------
# Simple linear beta schedule + q_sample (DDPM style)
# ------------------------
def get_linear_betas(num_timesteps, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, steps=num_timesteps)

class GaussianDiffusionSimple:
    def __init__(self, num_timesteps=500, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = get_linear_betas(num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)  # \bar{\alpha}_t

    def q_sample(self, x_start, t, noise):
        """
        x_start: [B, T, D]
        t: [B] integer timesteps in [0, num_timesteps-1]
        noise: same shape as x_start
        Returns x_t = sqrt(alpha_bar_t) * x_start + sqrt(1-alpha_bar_t) * noise
        """
        # gather alpha_bar for each t
        a_bar = self.alpha_bars[t].to(x_start.dtype)  # [B]
        a_bar = a_bar[:, None, None]                  # [B,1,1]
        return torch.sqrt(a_bar) * x_start + torch.sqrt(1.0 - a_bar) * noise

    def p_sample_loop(self, model, shape, cond, device, log_steps=5):
        B, T, D = shape
        x = torch.randn(shape, device=device)

        # choose timesteps to log
        log_indices = np.linspace(0, self.num_timesteps-1, log_steps, dtype=int)
        traj_snapshots = []
        
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((B,), t, dtype=torch.long, device=device)
            eps_theta = model(x, t_batch, cond)  # predicted noise

            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            beta_t = self.betas[t]

            # DDPM formula for x_{t-1}
            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
            x_prev = coef1 * (x - coef2 * eps_theta)

            # Add noise except for last step
            if t > 0:
                sigma_t = torch.sqrt(beta_t)
                noise = torch.randn_like(x)
                x_prev = x_prev + sigma_t * noise

            x = x_prev

            if t in log_indices:
                traj_snapshots.append(x.cpu().numpy())

        return x, traj_snapshots


@torch.no_grad()
def compute_validation_loss(model, dataloader, diffusion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        traj = batch["trajectory"].to(device)
        start = batch["start_state"].to(device)
        goal = batch["goal"].to(device)
        obstacles = batch["obstacles"].to(device)
        obs_mask = batch["obs_mask"].to(device)

        B, T, D = traj.shape
        obs_flat = obstacles.view(B, -1)
        cond = torch.cat([start, goal, obs_flat, obs_mask], dim=-1)

        t_batch = torch.randint(0, diffusion.num_timesteps, (B,), device=device)
        noise = torch.randn_like(traj)
        x_noisy = diffusion.q_sample(traj, t_batch, noise)

        pred_noise = model(x_noisy, t_batch, cond)
        loss = ((pred_noise - noise) ** 2).mean()

        total_loss += loss.item()
        n_batches += 1

    model.train()
    return total_loss / max(1, n_batches)


# ------------------------
# Training loop with checkpointing & resume
# ------------------------
def train_loop(model, dataloader, val_loader, norm_stats, optimizer, diffusion, args, device):
    train_losses = []
    val_losses = []

    # resume logic
    ckpt_dir = os.path.join("checkpoints", args.exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
    start_step = 0
    if len(ckpt_files) > 0:
        latest = ckpt_files[-1]
        ckpt = torch.load(os.path.join(ckpt_dir, latest), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0) + 1
        print(f"Resuming from checkpoint {latest} at step {start_step}")

    model.train()
    step = start_step
    save_every = args.save_every

    while step < args.num_steps:
        for batch in dataloader:
            traj = batch["trajectory"].to(device)        # [B, T, D]
            start = batch["start_state"].to(device)           # [B, start_dim]
            goal = batch["goal"].to(device)             # [B, goal_dim]
            obstacles = batch["obstacles"].to(device)   # [B, k_max, 4]
            obs_mask = batch["obs_mask"].to(device)     # [B, k_max]

            B, T, D = traj.shape

            # build conditioning vector: [start, goal, flattened obstacles, obs_mask]
            obs_flat = obstacles.view(B, -1)            # [B, k_max*4]
            cond = torch.cat([start, goal, obs_flat, obs_mask], dim=-1)  # [B, cond_dim]

            # sample diffusion timestep and noise
            t_batch = torch.randint(0, diffusion.num_timesteps, (B,), device=device)
            noise = torch.randn_like(traj)
            x_noisy = diffusion.q_sample(traj, t_batch, noise)

            pred_noise = model(x_noisy, t_batch, cond)
            loss = ((pred_noise - noise) ** 2).mean()
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log(
                {
                    "train/loss": loss.item(),
                    "diffusion/timestep_mean": t_batch.float().mean().item(),
                    "diffusion/noise_std": noise.std().item(),
                },
                step=step,
            )


            if step % save_every == 0:
                val_loss = compute_validation_loss(model, val_loader, diffusion, device)
                val_losses.append(val_loss)

                wandb.log(
                    {
                        "val/loss": val_loss,
                    },
                    step=step,
                )

                ckpt_path = os.path.join(ckpt_dir, f"ckpt_step{step}.pt")
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                }, ckpt_path)
                print(f"[Step {step}] train={loss.item():.6f}, val={val_loss:.6f}")

                # save a sample generation
                model.eval()
                with torch.no_grad():
                    # pick a random batch index
                    batch_idx = np.random.randint(len(val_loader))
                    val_iter = iter(val_loader)
                    for _ in range(batch_idx + 1):
                        val_batch = next(val_iter)  # advance iterator to random batch

                    # pick a random sample in that batch
                    idx = np.random.randint(val_batch["trajectory"].shape[0])

                    traj = val_batch["trajectory"][idx].unsqueeze(0).to(device)
                    start = val_batch["start_state"][idx].unsqueeze(0).to(device)
                    goal = val_batch["goal"][idx].unsqueeze(0).to(device)
                    obstacles = val_batch["obstacles"][idx].unsqueeze(0).to(device)
                    obs_mask = val_batch["obs_mask"][idx].unsqueeze(0).to(device)

                    B, T, D = traj.shape
                    obs_flat = obstacles.view(B, -1)
                    samp_cond = torch.cat([start, goal, obs_flat, obs_mask], dim=-1)[:1]
                    
                    gt_traj = traj[:1]
                    sample, traj_snapshots = diffusion.p_sample_loop(model, (1, T, D), samp_cond, device, log_steps=5)

                    b = 0  # pick first batch element
                    obs = obstacles[b]        # [k_max, 4]
                    mask = obs_mask[b].bool() # [k_max]
                    obs_valid = obs[mask]     # [N, 4], where N ≤ k_max
                    obs_pos = obs_valid[:, 0:2].cpu().numpy()* norm_stats["obs_std"][0:2] + norm_stats["obs_mean"][0:2]  # [N, 2]
                    obs_vel = obs_valid[:, 2:4].cpu().numpy()* norm_stats["obs_std"][2:4] + norm_stats["obs_mean"][2:4]  # [N, 2]

                    if obs_pos.shape[0] == 0:
                        obs_pos = None
                        obs_vel = None

                    start_np = start[b].cpu().numpy() * norm_stats["start_std"] + norm_stats["start_mean"]
                    goal_np  = goal[b].cpu().numpy() * norm_stats["goal_std"] + norm_stats["goal_mean"]

                    gt_traj = gt_traj 
                    sample = sample

                    log_trajectory_scene(
                        gt=gt_traj[0].cpu().numpy() * norm_stats["traj_std"] + norm_stats["traj_mean"],
                        pred=sample[0].cpu().numpy() * norm_stats["traj_std"] + norm_stats["traj_mean"],
                        start=start_np,
                        goal=goal_np,
                        obs_pos=obs_pos,
                        obs_vel=obs_vel,
                        step=step,
                        prefix="samples",
                    )


                    sample_denorm = sample * torch.from_numpy(norm_stats["traj_std"]).to(sample.device) + torch.from_numpy(norm_stats["traj_mean"]).to(sample.device)
                    gt_denorm     = gt_traj * torch.from_numpy(norm_stats["traj_std"]).to(gt_traj.device) + torch.from_numpy(norm_stats["traj_mean"]).to(gt_traj.device)

                    pos_err = torch.norm(sample_denorm - gt_denorm, dim=-1)  # (B,T)

                    ADE = pos_err.mean().item()
                    FDE = pos_err[:, -1].mean().item()

                    dt = 0.1  # seconds between waypoints
                    # Velocity
                    vel = (sample_denorm[:, 1:] - sample_denorm[:, :-1]) / dt   # (B,T-1,2)

                    # Acceleration
                    acc = (vel[:, 1:] - vel[:, :-1]) / dt                       # (B,T-2,2)

                    # Smoothness scalars
                    vel_mag = torch.norm(vel, dim=-1)   # (B,T-1)
                    acc_mag = torch.norm(acc, dim=-1)   # (B,T-2)

                    vel_smoothness = vel_mag.var(dim=1).mean().item()
                    acc_smoothness = acc_mag.mean().item()   # mean jerk proxy

                    wandb.log(
                        {
                            "sample/ADE": ADE,
                            "sample/FDE": FDE,
                            "sample/vel_variance": vel_smoothness,
                            "sample/mean_acceleration": acc_smoothness,
                        },
                        step=step,
                    )


                    sample_denorm_np = (sample[0].cpu().numpy() * norm_stats["traj_std"] + norm_stats["traj_mean"])
                    gt_denorm_np     = (gt_traj[0].cpu().numpy() * norm_stats["traj_std"] + norm_stats["traj_mean"])

                    # Log denoising snapshots as line plots
                    import matplotlib.pyplot as plt
                    import io

                    fig, ax = plt.subplots()
                    colors = plt.cm.viridis(np.linspace(0,1,len(traj_snapshots)))
                    for i, snap in enumerate(traj_snapshots):
                        traj = snap[0] * norm_stats["traj_std"] + norm_stats["traj_mean"]
                        ax.plot(traj[:,0], traj[:,1], color=colors[i], label=f't={i}')
                    ax.plot(gt_denorm_np[:,0], gt_denorm_np[:,1], 'k--', label='GT')
                    ax.set_title("Denoising snapshots")
                    ax.legend()
                    wandb.log({"sample/denoising_trajectory": wandb.Image(fig)}, step=step)
                    plt.close(fig)

                    
                    np.save(os.path.join(ckpt_dir, f"sample_step{step}.npy"), sample.cpu().numpy())
                model.train()

            step += 1
            if step >= args.num_steps:
                break

def compute_normalization_stats(dataset):
    """
    Compute mean and std for start, goal, trajectory, and obstacles.
    """
    start_list = []
    goal_list = []
    traj_list = []
    obs_list = []

    for i in range(len(dataset)):
        sample = dataset[i]
        start_list.append(sample["start_state"].numpy())
        goal_list.append(sample["goal"].numpy())
        traj_list.append(sample["trajectory"].numpy())

        obs = sample["obstacles"].numpy()
        mask = sample["obs_mask"].numpy().astype(bool)

        if mask.any():
            obs_list.append(obs[mask])   # ONLY real obstacles

    start_array = np.stack(start_list, axis=0)
    goal_array = np.stack(goal_list, axis=0)
    traj_array = np.concatenate(traj_list, axis=0)
    
    if len(obs_list) > 0:
        obs_array = np.concatenate(obs_list, axis=0)
        obs_mean = obs_array.mean(axis=0)
        obs_std  = obs_array.std(axis=0) + 1e-8
    else:
        obs_mean = np.zeros(4)
        obs_std  = np.ones(4)

    stats = {
        "start_mean": start_array.mean(axis=0),
        "start_std":  start_array.std(axis=0) + 1e-8,
        "goal_mean":  goal_array.mean(axis=0),
        "goal_std":   goal_array.std(axis=0) + 1e-8,
        "traj_mean":  traj_array.mean(axis=0),
        "traj_std":   traj_array.std(axis=0) + 1e-8,
        "obs_mean":   obs_mean,
        "obs_std":    obs_std,
    }
    return stats

# ------------------------
# Main: argparsing and setup
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="diffusion_run4")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=100000)
    parser.add_argument("--num_diffusion_timesteps", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=16, help="trajectory length per sample")
    parser.add_argument("--k_max", type=int, default=10, help="max number of dynamic obstacles")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--hidden_dim", type=int, default=256)
    args = parser.parse_args()

    wandb.init(
        project="Social Nav Diffusion",
        name=args.exp_name,
        config=vars(args),   # logs all hyperparameters
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    all_files = sorted(glob.glob(os.path.join(args.dataset_dir, "*.npz"))
                   + glob.glob(os.path.join(args.dataset_dir, "*.npy")))

    # ---- train/val split (e.g. 80/20) ----
    n_total = len(all_files)
    n_val = int(0.2 * n_total)

    train_files = all_files[:-n_val]
    val_files   = all_files[-n_val:]

    # temp dataset to compute normalization
    train_ds_tmp = RobotTrajectoryDataset(
        args.dataset_dir, args.horizon, args.k_max, files=train_files
    )

    norm_stats = compute_normalization_stats(train_ds_tmp)
    print("Normalization stats:", norm_stats)
    ckpt_dir = f"checkpoints/{args.exp_name}"
    os.makedirs(ckpt_dir, exist_ok=True)

    np.save(os.path.join(ckpt_dir, "norm_stats.npy"), norm_stats)

    train_ds = RobotTrajectoryDataset(
        args.dataset_dir, args.horizon, args.k_max,
        norm_stats=norm_stats, files=train_files
    )
    val_ds = RobotTrajectoryDataset(
        args.dataset_dir, args.horizon, args.k_max,
        norm_stats=norm_stats, files=val_files
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, drop_last=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, drop_last=False, num_workers=4
    )

    # infer dims from dataset first file
    sample = train_ds_tmp[0]
    traj_dim = sample["trajectory"].shape[1]
    start_dim = sample["start_state"].shape[0]
    goal_dim = sample["goal"].shape[0]
    cond_dim = start_dim + goal_dim + args.k_max * 4 + args.k_max  # start + goal + obstacles flat + mask

    model = ConditionalTemporalNet(traj_dim, cond_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    diffusion = GaussianDiffusionSimple(device=device)

    wandb.watch(
        model,
        log="all",      # gradients + weights
        log_freq=500    # don’t do every step (too slow)
    )

    # dummy inputs
    B, T, D = 1, args.horizon, traj_dim
    dummy_x = torch.randn(B, T, D).to(device)
    dummy_t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)
    dummy_cond = torch.randn(B, cond_dim).to(device)

    y = model(dummy_x, dummy_t, dummy_cond)

    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.format = "png"
    dot.render("model_graph")
    wandb.log({"model/graph": wandb.Image("model_graph.png")})


    print("traj_dim", traj_dim, "cond_dim", cond_dim, "horizon", args.horizon, "k_max", args.k_max)
    
    train_loop(model, train_loader, val_loader, norm_stats, optimizer, diffusion, args, device)

if __name__ == "__main__":
    main()
