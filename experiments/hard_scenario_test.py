# save as hard_scenario_test.py
# Tests warm-start ONLY on scenarios where cold-start took >100ms
# These are the cases where warm-starting matters most

import pandas as pd
import numpy as np

# load your existing results
df = pd.read_csv("results/warmstart_L1_v3_33bus.csv")

# filter to hard cases only
hard = df[df['cold_ms'] > 100]
easy = df[df['cold_ms'] <= 100]

print(f"Hard scenarios (cold > 100ms): {len(hard)}")
print(f"Easy scenarios (cold ≤ 100ms): {len(easy)}")
print()

if len(hard) > 0:
    print("HARD SCENARIOS ONLY:")
    print(f"  Cold mean:  {hard['cold_ms'].mean():.1f} ms")
    print(f"  Warm mean:  {hard['effective_warm'].mean():.1f} ms")
    print(f"  Speedup:    {hard['cold_ms'].mean()/hard['effective_warm'].mean():.3f}x")
    print(f"  Fallback:   {hard['fallback'].mean()*100:.1f}%")
    print()

print("EASY SCENARIOS ONLY:")
print(f"  Cold mean:  {easy['cold_ms'].mean():.1f} ms")
print(f"  Warm mean:  {easy['effective_warm'].mean():.1f} ms")
print(f"  Speedup:    {easy['cold_ms'].mean()/easy['effective_warm'].mean():.3f}x")