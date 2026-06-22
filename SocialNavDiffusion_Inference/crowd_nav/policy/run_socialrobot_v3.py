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
import torch.nn.functional as F

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

class ConditionalTemporalNet(nn.Module):
    def __init__(
        self,
        traj_dim,            # K (trajectory length)
        start_dim,          # e.g. velocity magnitude or start state dim
        goal_dim,           # usually 2
        k_max,              # max number of obstacles
        obs_dim=4,          # (x,y,vx,vy)
        d_model=64,
        n_layers=2,
        n_heads=2,
    ):
        super().__init__()
        self.traj_dim = traj_dim
        self.d_model = d_model
        self.k_max = k_max

        # -------- Trajectory embedding --------
        self.traj_embed = nn.Linear(2, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, traj_dim, d_model))
        self.pos_scale = nn.Parameter(torch.tensor(0.1))

        # -------- Timestep embedding --------
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

        # -------- Conditioning embeddings (tokens) --------
        self.start_embed = nn.Linear(start_dim, d_model)
        self.goal_embed = nn.Linear(goal_dim, d_model)
        self.obs_embed = nn.Linear(obs_dim, d_model)
        

        # -------- Transformer blocks --------
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True),
                "cross_attn": nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True),
                "ff": nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model)
                ),
                "ln1": nn.LayerNorm(d_model),
                "ln2": nn.LayerNorm(d_model),
                "ln3": nn.LayerNorm(d_model),
            }) for _ in range(n_layers)
        ])

        # -------- Output head (predict noise) --------
        self.out = nn.Linear(d_model, 2)

    def forward(self, tau, t, start, goal, obstacles, obs_mask):
        """
        tau:        (B, K, 2)
        t:          (B,)
        start:      (B, start_dim)
        goal:       (B, goal_dim)
        obstacles:  (B, k_max, obs_dim)
        obs_mask:   (B, k_max)   1 = valid, 0 = padded
        """

        B, K, _ = tau.shape

        # -------- Trajectory tokens --------
        x = self.traj_embed(tau)                    # (B,K,d)
        x = x + self.pos_scale * self.pos_embed[:, :K]               # positional encoding

        # -------- Timestep conditioning --------
        t_emb = self.sinusoidal_embedding(t, self.d_model)
        t_emb = self.time_mlp(t_emb)                # (B,d)
        x = x + t_emb[:, None, :]                   # broadcast over K

        # -------- Conditioning tokens --------
        start_tok = self.start_embed(start)[:, None, :]      # (B,1,d)
        goal_tok  = self.goal_embed(goal)[:, None, :]        # (B,1,d)

        # ---- Roll obstacle positions forward in time ----
        dt = 0.1  # We assume 0.1 s between trajectory waypoints
        times = torch.arange(K, device=tau.device).float() * dt  # (K,)
        obs_xy = obstacles[..., :2]     # (B, k_max, 2)
        obs_vel = obstacles[..., 2:]    # (B, k_max, 2)
        # Rollout: (B, K, k_max, 2)
        obs_xy_t = obs_xy[:, None] + times[None, :, None, None] * obs_vel[:, None]
        # tau: (B, K, 2)
        tau_xy = tau[:, :, None, :]     # (B, K, 1, 2)
        rel_obs = obs_xy_t - tau_xy     # (B, K, k_max, 2)
        obs_feat = torch.cat(
            [rel_obs, obs_vel[:, None].expand(-1, K, -1, -1)],
            dim=-1
        )  # (B, K, k_max, 4)
        # Mean-pool over time (masked)
        mask = obs_mask[:, None, :, None]  # (B, 1, k_max, 1)
        obs_feat = obs_feat * mask

        obs_feat_pooled = obs_feat.sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        # (B, k_max, 4)
        obs_tok   = self.obs_embed(obs_feat_pooled)                # (B,k_max,d)

        cond_tokens = torch.cat(
            [start_tok, goal_tok, obs_tok],
            dim=1
        )  # (B, 2 + k_max, d)

        # -------- Conditioning mask --------
        cond_mask = torch.cat(
            [
                torch.ones(B, 2, device=tau.device),
                obs_mask
            ],
            dim=1
        ).bool()  # True = keep, False = mask

        # -------- Transformer --------
        for block in self.blocks:
            # Self-attention over trajectory
            x_res, _ = block["self_attn"](x, x, x)
            x = block["ln1"](x + x_res)

            # Cross-attention to conditioning tokens
            x_res, _ = block["cross_attn"](
                x, cond_tokens, cond_tokens,
                key_padding_mask=~cond_mask
            )
            x = block["ln2"](x + x_res)

            # Feed-forward
            x = block["ln3"](x + block["ff"](x))

        return self.out(x)  # (B,K,2)

    @staticmethod
    def sinusoidal_embedding(t, dim):
        """
        t: (B,)
        returns: (B, dim)
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


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

    def p_sample_loop(self, model, shape, start, goal, obstacles, obs_mask, device):
        B, T, D = shape
        x = torch.randn(shape, device=device)
        
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((B,), t, dtype=torch.long, device=device)
            eps_theta = model(x, t_batch, start, goal, obstacles, obs_mask)  # predicted noise

            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            beta_t = self.betas[t]

            # --- DDPM posterior mean ---
            mean = (
                1.0 / torch.sqrt(alpha_t)
            ) * (
                x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_theta
            )

            # --- Sample x_{t-1} ---
            if t > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean

        return x

    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        shape,
        start,
        goal,
        obstacles,
        obs_mask,
        device,
        steps=20,
        eta=0.0
    ):
        B, T, D = shape
        x = torch.randn(shape, device=device)

        times = torch.linspace(
            self.num_timesteps - 1, 0, steps, device=device
        ).long()

        for i in range(len(times) - 1):
            t = times[i]
            t_next = times[i + 1]
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            eps = model(x, t_batch, start, goal, obstacles, obs_mask)

            alpha_bar = self.alpha_bars[t]
            alpha_bar_next = self.alpha_bars[t_next]

            x0 = (x - torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha_bar)

            c1 = eta * torch.sqrt(
                (1 - alpha_bar / alpha_bar_next) *
                (1 - alpha_bar_next) / (1 - alpha_bar)
            )
            c2 = torch.sqrt(1 - alpha_bar_next - c1**2)

            noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
            x = torch.sqrt(alpha_bar_next) * x0 + c2 * eps + c1 * noise

        return x



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

        # t_batch = torch.randint(0, diffusion.num_timesteps, (B,), device=device)
        # bias towards larger t
        t_batch = torch.randint(
            diffusion.num_timesteps // 4,
            diffusion.num_timesteps,
            (B,),
            device=device
        )

        noise = torch.randn_like(traj)
        x_noisy = diffusion.q_sample(traj, t_batch, noise)

        pred_noise = model(x_noisy, t_batch, start, goal, obstacles, obs_mask)
        loss = ((pred_noise - noise) ** 2).mean()

        total_loss += loss.item()
        n_batches += 1

    model.train()
    return total_loss / max(1, n_batches)


def compute_eval_loss(model, dataloader, diffusion, device, max_batches=10):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            traj = batch["trajectory"].to(device)
            start = batch["start_state"].to(device)
            goal = batch["goal"].to(device)
            obstacles = batch["obstacles"].to(device)
            obs_mask = batch["obs_mask"].to(device)

            B, T, D = traj.shape
            t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)
            noise = torch.randn_like(traj)
            x_noisy = diffusion.q_sample(traj, t, noise)
            pred = model(x_noisy, t, start, goal, obstacles, obs_mask)
            loss = ((pred - noise) ** 2).mean()

            total += loss.item()
            n += 1
    model.train()
    return total / max(1, n)


# ------------------------
# Training loop with checkpointing & resume
# ------------------------
def train_loop(model, dataloader, train_eval_loader, val_loader, optimizer, diffusion, args, device):
    train_losses = []
    val_losses = []
    train_avg_losses = []

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

            # sample diffusion timestep and noise
            # t_batch = torch.randint(0, diffusion.num_timesteps, (B,), device=device)
            # bias towards larger t
            t_batch = torch.randint(
                diffusion.num_timesteps // 4,
                diffusion.num_timesteps,
                (B,),
                device=device
            )
            noise = torch.randn_like(traj)
            x_noisy = diffusion.q_sample(traj, t_batch, noise)

            pred_noise = model(x_noisy, t_batch, start, goal, obstacles, obs_mask)
            loss = ((pred_noise - noise) ** 2).mean()
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % save_every == 0:
                train_loss_avg = compute_eval_loss(model, train_eval_loader, diffusion, device, max_batches=10)
                val_loss = compute_validation_loss(model, val_loader, diffusion, device)
                val_losses.append(val_loss)
                train_avg_losses.append(train_loss_avg)
                ckpt_path = os.path.join(ckpt_dir, f"ckpt_step{step}.pt")
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_losses": train_losses, #per-batch
                    "train_loss_avg": train_avg_losses,    # full-dataset
                    "val_losses": val_losses,
                }, ckpt_path)
                # print(f"[Step {step}] train={loss.item():.6f}, val={val_loss:.6f}")
                print(f"[Step {step}] train(batch)={loss.item():.6f}, train(avg)={train_loss_avg:.6f}, val={val_loss:.6f}")

                # save a sample generation
                model.eval()
                with torch.no_grad():
                    # samp_cond = cond[:1]  # first example condition
                    sample = diffusion.p_sample_loop(model, (1, T, D), start[:1], goal[:1], obstacles[:1], obs_mask[:1], device)
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
    parser.add_argument("--hidden_dim", type=int, default=128)
    args = parser.parse_args()

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

    train_eval_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2
    )


    # infer dims from dataset first file
    sample = train_ds_tmp[0]
    traj_dim = sample["trajectory"].shape[0]
    start_dim = sample["start_state"].shape[0]
    goal_dim = sample["goal"].shape[0]
    cond_dim = start_dim + goal_dim + args.k_max * 4 + args.k_max  # start + goal + obstacles flat + mask

    model = ConditionalTemporalNet(traj_dim, start_dim, goal_dim, args.k_max).to(device)
    nn.init.xavier_uniform_(model.out.weight)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    diffusion = GaussianDiffusionSimple(device=device)

    print("traj_dim", traj_dim, "cond_dim", cond_dim, "horizon", args.horizon, "k_max", args.k_max)
    
    train_loop(model, train_loader, train_eval_loader, val_loader, optimizer, diffusion, args, device)

if __name__ == "__main__":
    main()
