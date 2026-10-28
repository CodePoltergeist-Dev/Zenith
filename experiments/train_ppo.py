import gymnasium as gym
import gym_power_res
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
import os
import time

try:
    import tensorboard  # noqa: F401
    TENSORBOARD_LOG = "logs/tensorboard/"
except ImportError:
    TENSORBOARD_LOG = None

# ── Directories ───────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("eval_logs", exist_ok=True)

# ── Environment ───────────────────────────────────────────────
train_env = Monitor(gym.make("RestorationDisEnv-v1"), "logs/train")
eval_env  = Monitor(gym.make("RestorationDisEnv-v1"), "logs/eval")

# ── Callback: prints a summary every N episodes ───────────────
class TrainingMonitor(BaseCallback):
    def __init__(self, print_every=50):
        super().__init__()
        self.print_every = print_every
        self.episode_rewards = []
        self.episode_count = 0
        self.optimal_actions = 0
        self.total_actions = 0
        self.start_time = time.time()

    def _on_step(self):
        infos = self.locals.get("infos", [{}])
        for info in infos:
            status = info.get("status")
            if status:
                self.total_actions += 1
                if status == "optimal":
                    self.optimal_actions += 1
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_count += 1
                if self.episode_count % self.print_every == 0:
                    recent = self.episode_rewards[-self.print_every:]
                    opt_rate = self.optimal_actions / max(self.total_actions, 1) * 100
                    elapsed = (time.time() - self.start_time) / 60
                    print(
                        f"Episode {self.episode_count:4d} | "
                        f"Mean reward: {np.mean(recent):8.2f} | "
                        f"Optimal rate: {opt_rate:5.1f}% | "
                        f"Elapsed: {elapsed:.1f}min"
                    )
        return True

# ── Model ─────────────────────────────────────────────────────
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
    verbose=0,
    tensorboard_log=TENSORBOARD_LOG,
    policy_kwargs=dict(net_arch=[128, 128])
)

print("=" * 55)
print("PPO Training — 33-bus RestorationDisEnv")
print("=" * 55)
print(f"Observation space: {train_env.observation_space.shape}")
print(f"Action space:      {train_env.action_space.n} actions")
print(f"Network arch:      [128, 128]")
print(f"Total timesteps:   100,000")
print("=" * 55)
print()

# ── Eval callback: saves best model automatically ─────────────
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="models/",
    log_path="eval_logs/",
    eval_freq=2000,
    n_eval_episodes=10,
    deterministic=True,
    verbose=0
)

monitor = TrainingMonitor(print_every=50)

# ── Train ─────────────────────────────────────────────────────
start = time.time()
model.learn(
    total_timesteps=100_000,
    callback=[monitor, eval_callback],
    progress_bar=False
)
elapsed = time.time() - start

# ── Save final model ──────────────────────────────────────────
model.save("models/ppo_33bus_final")
print(f"\nTraining complete in {elapsed/60:.1f} minutes")
print(f"Final model saved → models/ppo_33bus_final.zip")
print(f"Best model saved  → models/best_model.zip")

# ── Quick eval of final model ─────────────────────────────────
print("\nEvaluating final model over 20 episodes...")
test_env = gym.make("RestorationDisEnv-v1")
rewards = []
optimal_rates = []

for ep in range(20):
    obs, _ = test_env.reset()
    done = False
    ep_reward = 0
    optimal_count = 0
    total_steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = test_env.step(int(action))
        ep_reward += reward
        total_steps += 1
        if info.get("status") == "optimal":
            optimal_count += 1
        done = term or trunc

    rewards.append(ep_reward)
    optimal_rates.append(optimal_count / total_steps * 100)

test_env.close()
train_env.close()
eval_env.close()

print(f"\n{'='*55}")
print("TRAINED PPO — 33-bus evaluation")
print(f"{'='*55}")
print(f"Mean reward:        {np.mean(rewards):.2f}")
print(f"Std reward:         {np.std(rewards):.2f}")
print(f"Best episode:       {max(rewards):.2f}")
print(f"Mean optimal rate:  {np.mean(optimal_rates):.1f}%")
print(f"{'='*55}")
print("\nBaseline comparison:")
print(f"  Random policy mean reward:  -292.58")
print(f"  Trained PPO mean reward:    {np.mean(rewards):.2f}")
print(f"  Random optimal rate:        62.5%")
print(f"  Trained optimal rate:       {np.mean(optimal_rates):.1f}%")
print(f"\nIf PPO mean reward > -292.58 and optimal rate > 62.5%, training worked.")