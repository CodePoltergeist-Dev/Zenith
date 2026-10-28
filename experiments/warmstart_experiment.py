import gymnasium as gym
import gym_power_res
from stable_baselines3 import PPO
import numpy as np
import time
import csv
import os
import pyomo.environ as pyo
import torch

os.makedirs("results", exist_ok=True)

model = PPO.load("models/best_model")
env = gym.make("RestorationDisEnv119-v1")

# The saved PPO policy was trained on the 33-bus observation space.
# Use the same environment family to avoid a shape mismatch at inference time.
def build_matching_env(model_obj):
    candidates = ["RestorationDisEnv-v1", "RestorationDisEnv119-v1"]
    for env_id in candidates:
        try:
            candidate = gym.make(env_id)
        except Exception:
            continue

        obs_match = candidate.observation_space.shape == model_obj.observation_space.shape
        action_match = getattr(candidate.action_space, "n", None) == getattr(model_obj.action_space, "n", None)
        if obs_match and action_match:
            return candidate, env_id

        candidate.close()

    raise ValueError(
        f"No compatible Gym environment found for model obs={model_obj.observation_space.shape}, "
        f"actions={getattr(model_obj.action_space, 'n', None)}. "
        f"Available candidates: {candidates}"
    )


env, env_id = build_matching_env(model)
print(f"Loaded PPO model from models/v2/best_model; using {env_id} to match observation/action dimensions.")
raw_env = env.unwrapped

print("=" * 60)
print("Warm-Start Experiment v3 — Level 1, 33-bus")
print("Fix 1: solve_ms from info dict")
print("Fix 2: mixed policy (PPO + forced tie-line actions)")
print("=" * 60)

def get_ppo_probs(obs):
    obs_t = model.policy.obs_to_tensor(obs)[0]
    with torch.no_grad():
        dist  = model.policy.get_distribution(obs_t)
        probs = dist.distribution.probs.cpu().numpy().flatten()
    return probs

def inject_hints(raw_env, probs):
    """
    Inject PPO probability vector as initial variable hints
    into the current Pyomo model before warm-start solve.
    """
    try:
        m        = raw_env.sim_case.model
        tielines = list(raw_env.ppc['tieline'])
        injected = 0
        predicted_action = int(np.argmax(probs))

        for i, tl in enumerate(tielines):
            # hint=1 if PPO predicts this tie-line should close
            hint = 1 if predicted_action == (i + 1) else 0
            for varname in ['x', 'xL', 'alpha']:
                try:
                    var = getattr(m, varname)
                    for t in range(1, len(tielines) + 2):
                        try:
                            var[tl, t].set_value(hint)
                            injected += 1
                        except (KeyError, ValueError):
                            pass
                except AttributeError:
                    pass
        return injected
    except Exception:
        return 0

def solve_warm(raw_env):
    try:
        opt = pyo.SolverFactory("appsi_highs")
        t0  = time.perf_counter()
        res = opt.solve(
            raw_env.sim_case.model,
            tee=False,
            load_solutions=False,
            warmstart=True
        )
        warm_ms = (time.perf_counter() - t0) * 1000
        status  = str(res.solver.termination_condition)
        return warm_ms, status
    except Exception as e:
        print("Warm solve failed:", repr(e))
        return None, "error"

# ── Experiment ────────────────────────────────────────────────
# Use BOTH ppo actions AND forced tie-line actions
# so we get warm-start data on non-trivial switch predictions
N_EPISODES  = 50
results     = []
cold_all    = []
warm_all    = []
fallbacks   = 0

print(f"\n{'Ep':>3} {'Stp':>3} {'Act':>4} {'Cold':>8} "
      f"{'Warm':>8} {'Spdup':>7} {'FB':>4} {'Src':>5}")
print("-" * 55)

for ep in range(N_EPISODES):
    obs, info = env.reset()
    done      = False
    step_n    = 0

    while not done:
        step_n += 1
        probs = get_ppo_probs(obs)

        # Alternate: even episodes use PPO, odd use forced
        # tie-line action to get non-trivial warm-start data
        if ep % 2 == 0:
            action = int(np.argmax(probs))
            src    = "ppo"
        else:
            # force a random tie-line action (1-5) instead of no-op
            action = np.random.randint(1, env.action_space.n)
            # update probs to reflect forced action for hint injection
            probs  = np.zeros(env.action_space.n)
            probs[action] = 1.0
            src    = "forced"

        # ── 1. Inject hints into current model ────────────────
        hints = inject_hints(raw_env, probs)

        # ── 2. Warm solve on current problem state ────────────
        warm_ms, ws_status = solve_warm(raw_env)

        # ── 3. Cold solve via env.step ────────────────────────
        obs, reward, term, trunc, info = env.step(action)

        # FIX 1: get solve_ms from info dict directly
        cold_ms = info.get('solve_ms', None)
        done    = term or trunc

        if cold_ms is None or warm_ms is None:
            print(
                f"DEBUG: cold_ms={cold_ms}, warm_ms={warm_ms}, "
                f"ws_status={ws_status}, info_keys={list(info.keys())}, info={info}"
            )
            continue

        # ── 4. Fallback check ─────────────────────────────────
        ws_ok       = ws_status in ["optimal", "feasible"]
        not_worse   = warm_ms <= cold_ms * 1.5
        is_fallback = not (ws_ok and not_worse)

        if is_fallback:
            fallbacks += 1
            effective_warm = cold_ms
        else:
            effective_warm = warm_ms

        speedup = cold_ms / effective_warm if effective_warm > 0 else 1.0
        cold_all.append(cold_ms)
        warm_all.append(effective_warm)

        fb = "YES" if is_fallback else "-"
        print(f"{ep+1:>3} {step_n:>3} {action:>4} "
              f"{cold_ms:>8.1f} {effective_warm:>8.1f} "
              f"{speedup:>7.2f}x {fb:>4} {src:>5}")

        results.append({
            "episode":        ep + 1,
            "step":           step_n,
            "action":         action,
            "action_src":     src,
            "cold_ms":        round(cold_ms, 3),
            "warm_ms":        round(warm_ms, 3),
            "effective_warm": round(effective_warm, 3),
            "speedup":        round(speedup, 3),
            "hints_injected": hints,
            "ws_status":      ws_status,
            "env_status":     info.get("status", ""),
            "fallback":       is_fallback,
        })

# ── Summary ───────────────────────────────────────────────────
total    = len(cold_all)
speedups = [c/w for c, w in zip(cold_all, warm_all) if w > 0]

print("\n" + "=" * 60)
print(f"RESULTS — Level 1 Warm-Start, 33-bus, {N_EPISODES} episode(s)")
print("=" * 60)
print(f"Total solve pairs:       {total}")
if total:
    print(f"Fallbacks:               {fallbacks} ({fallbacks/total*100:.1f}%)")
else:
    print(f"Fallbacks:               {fallbacks} (n/a)")
print()
print(f"{'Metric':<28} {'Cold':>9} {'Warm':>9}")
print("-" * 48)
if total:
    print(f"{'Mean (ms)':<28} {np.mean(cold_all):>9.2f} {np.mean(warm_all):>9.2f}")
    print(f"{'Std (ms)':<28} {np.std(cold_all):>9.2f}  {np.std(warm_all):>9.2f}")
    print(f"{'Min (ms)':<28} {np.min(cold_all):>9.2f}  {np.min(warm_all):>9.2f}")
    print(f"{'Max (ms)':<28} {np.max(cold_all):>9.2f}  {np.max(warm_all):>9.2f}")
    print()
    print(f"Mean speedup:            {np.mean(speedups):.3f}x")
    print(f"Median speedup:          {np.median(speedups):.3f}x")
    helped = sum(s > 1.05 for s in speedups)
    print(f"Helped (>1.05x):         {helped}/{total} ({helped/total*100:.1f}%)")
    print()

# Split by action source
ppo_rows    = [r for r in results if r['action_src'] == 'ppo']
forced_rows = [r for r in results if r['action_src'] == 'forced']
if ppo_rows:
    ppo_speedups = [r['speedup'] for r in ppo_rows]
    print(f"PPO actions mean speedup:    {np.mean(ppo_speedups):.3f}x")
if forced_rows:
    forced_speedups = [r['speedup'] for r in forced_rows]
    print(f"Forced actions mean speedup: {np.mean(forced_speedups):.3f}x")

print()
print(f"Phase 3 cold mean (ref):  52.33 ms")
print(f"This run cold mean:       {np.mean(cold_all):.2f} ms" if total else "This run cold mean:       n/a")
print("=" * 60)

with open("results/warmstart_L1_119bus.csv", "w", newline="") as f:
    if results:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

print(f"\nSaved → results/warmstart_L1_v3_33bus.csv")
env.close()