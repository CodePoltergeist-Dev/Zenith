# save as test_timing_119.py
import gymnasium as gym
import gym_power_res
import numpy as np
import csv
import time

env = gym.make("RestorationDisEnv119-v1")

solve_times = []
rewards = []

N_EPISODES = 20
print(f"Running {N_EPISODES} episodes on 119-bus...\n")

for ep in range(N_EPISODES):
    obs, info = env.reset()
    
    if hasattr(env.unwrapped, '_last_solve_ms'):
        solve_times.append(('reset', ep, env.unwrapped._last_solve_ms, info.get('status')))

    terminated = False
    truncated = False
    ep_reward = 0

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward

        if hasattr(env.unwrapped, '_last_solve_ms'):
            solve_times.append(('step', ep, env.unwrapped._last_solve_ms, info.get('status')))

    rewards.append(ep_reward)
    print(f"Episode {ep+1:02d}: reward={ep_reward:.1f}, steps={len([t for t in solve_times if t[1]==ep])}")

env.close()

times_ms = [t[2] for t in solve_times]
optimal_times = [t[2] for t in solve_times if t[3] == 'optimal']
infeasible_times = [t[2] for t in solve_times if t[3] == 'infeasible']

print("\n" + "=" * 50)
print("COLD-START MILP BASELINE — 119-bus, random policy")
print("=" * 50)
print(f"Total solves:          {len(times_ms)}")
print(f"Optimal solves:        {len(optimal_times)}")
print(f"Infeasible solves:     {len(infeasible_times)}")
print()

if times_ms:
    print(f"Mean solve time:       {np.mean(times_ms):.2f} ms")
    print(f"Std dev:               {np.std(times_ms):.2f} ms")
    print(f"Min solve time:        {np.min(times_ms):.2f} ms")
    print(f"Max solve time:        {np.max(times_ms):.2f} ms")
    print()
    if optimal_times:
        print(f"Optimal mean:          {np.mean(optimal_times):.2f} ms")
    if infeasible_times:
        print(f"Infeasible mean:       {np.mean(infeasible_times):.2f} ms")

print()
print(f"Mean episode reward:   {np.mean(rewards):.2f}")
print("=" * 50)

with open('baseline_timing_119bus.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['call_type', 'episode', 'solve_ms', 'status'])
    writer.writerows(solve_times)

print("\nSaved to baseline_timing_119bus.csv")
print("\nCompare with 33-bus mean of 52.33 ms — the ratio is your scalability evidence.")