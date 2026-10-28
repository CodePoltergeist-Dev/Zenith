# save as test_timing.py
import gymnasium as gym
import gym_power_res
import numpy as np

env = gym.make("RestorationDisEnv-v1")

solve_times = []
rewards = []
statuses = []

N_EPISODES = 20
print(f"Running {N_EPISODES} episodes to collect baseline solve times...\n")

for ep in range(N_EPISODES):
    obs, info = env.reset()
    
    # record reset solve time
    if hasattr(env.unwrapped, '_last_solve_ms'):
        solve_times.append(('reset', ep, env.unwrapped._last_solve_ms, info.get('status')))
    
    terminated = False
    truncated = False
    step = 0
    ep_reward = 0

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        ep_reward += reward

        # record step solve time
        if hasattr(env.unwrapped, '_last_solve_ms'):
            solve_times.append(('step', ep, env.unwrapped._last_solve_ms, info.get('status')))
        
        statuses.append(info.get('status'))

    rewards.append(ep_reward)

env.close()

# ── Analysis ──────────────────────────────────────────────────
times_ms = [t[2] for t in solve_times]
optimal_times = [t[2] for t in solve_times if t[3] == 'optimal']
infeasible_times = [t[2] for t in solve_times if t[3] == 'infeasible']

print("=" * 50)
print("COLD-START MILP BASELINE — 33-bus, random policy")
print("=" * 50)
print(f"Total solves:          {len(times_ms)}")
print(f"Optimal solves:        {len(optimal_times)}")
print(f"Infeasible solves:     {len(infeasible_times)}")
print()
print(f"Mean solve time:       {np.mean(times_ms):.2f} ms")
print(f"Std dev:               {np.std(times_ms):.2f} ms")
print(f"Min solve time:        {np.min(times_ms):.2f} ms")
print(f"Max solve time:        {np.max(times_ms):.2f} ms")
print()
print(f"Optimal mean:          {np.mean(optimal_times):.2f} ms" if optimal_times else "No optimal solves")
print(f"Infeasible mean:       {np.mean(infeasible_times):.2f} ms" if infeasible_times else "No infeasible solves")
print()
print(f"Mean episode reward:   {np.mean(rewards):.2f}")
print("=" * 50)
print("\nThese are your BASELINE numbers — the timing your warm-start must beat.")

# save to CSV for your results table
import csv
with open('baseline_timing_33bus.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['call_type', 'episode', 'solve_ms', 'status'])
    writer.writerows(solve_times)

print("\nSaved to baseline_timing_33bus.csv")