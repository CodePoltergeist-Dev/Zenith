# save as retrain_ppo.py
import gymnasium as gym
import gym_power_res
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import os
import time

try:
    import tensorboard  # noqa: F401
    TENSORBOARD_LOG = "logs/tensorboard_v2/"
except ImportError:
    TENSORBOARD_LOG = None

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

train_env = Monitor(gym.make("RestorationDisEnv-v1"), "logs/train_v2")
eval_env  = Monitor(gym.make("RestorationDisEnv-v1"), "logs/eval_v2")

model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=0,
    tensorboard_log=TENSORBOARD_LOG,
    policy_kwargs=dict(net_arch=[128, 128])
)

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path="models/v2/",
    log_path="eval_logs/v2/",
    eval_freq=2000,
    n_eval_episodes=10,
    deterministic=True,
    verbose=0
)

print("Retraining PPO with entropy bonus (ent_coef=0.01)")
print("Target: agent should USE tie-line actions, not just no-op")
print()

from stable_baselines3.common.callbacks import BaseCallback

class ActionMonitor(BaseCallback):
    def __init__(self):
        super().__init__()
        self.action_counts = {}
        self.ep_count = 0
        self.ep_rewards = []

    def _on_step(self):
        actions = self.locals.get("actions", [])
        for a in actions:
            self.action_counts[int(a)] = self.action_counts.get(int(a), 0) + 1
        infos = self.locals.get("infos", [{}])
        for info in infos:
            if "episode" in info:
                self.ep_count += 1
                self.ep_rewards.append(info["episode"]["r"])
                if self.ep_count % 100 == 0:
                    total_acts = sum(self.action_counts.values())
                    noop_pct   = self.action_counts.get(0, 0) / total_acts * 100
                    recent_r   = np.mean(self.ep_rewards[-100:])
                    print(f"Ep {self.ep_count:4d} | "
                          f"Reward: {recent_r:7.1f} | "
                          f"No-op %: {noop_pct:5.1f}% | "
                          f"Action dist: { {k:v for k,v in sorted(self.action_counts.items())} }")
        return True

action_monitor = ActionMonitor()
start = time.time()

model.learn(
    total_timesteps=100_000,
    callback=[eval_cb, action_monitor],
    progress_bar=False
)

elapsed = (time.time() - start) / 60
model.save("models/v2/ppo_entropy_final")
print(f"\nDone in {elapsed:.1f} min")
print("Saved → models/v2/ppo_entropy_final.zip")
print("Best → models/v2/best_model.zip")

train_env.close()
eval_env.close()