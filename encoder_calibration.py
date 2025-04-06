import pandas as pd
import os
import numpy as np

# === CONFIG ===
# Key : (filename, real_cm, duration_ms, direction)
movement_data = {
    'a': ("datasets/filtered_datasets/move_a_17.7_kalman_output.csv", 17.7, 200, "left"),
    's': ("datasets/filtered_datasets/move_s_11.5_kalman_output.csv", 11.5, 150, "left"),
    'd': ("datasets/filtered_datasets/move_d_7.4_kalman_output.csv", 7.4, 100, "left"),
    'f': ("datasets/filtered_datasets/move_f_2.3_kalman_output.csv", 2.3, 50, "left"),
    'k': ("datasets/filtered_datasets/move_k_7.1_kalman_output.csv", 7.1, 100, "right"),
    'l': ("datasets/filtered_datasets/move_l_12.4_kalman_output.csv", 12.4, 150, "right"),
    ';': ("datasets/filtered_datasets/move_;_18_kalman_output.csv", 18.0, 200, "right"),
    # 'j' is excluded (too small)
}

# === COMPUTE RESULTS ===
results = []
for key, (file, real_cm, duration_ms, direction) in movement_data.items():
    if not os.path.exists(file):
        print(f"❌ File not found: {file}")
        continue

    df = pd.read_csv(file)
    encoder_col = df.columns[-1]
    ticks = df[encoder_col].values
    tick_diff = abs(ticks.max() - ticks.min())

    cm_per_tick = real_cm / tick_diff
    ticks_per_cm = tick_diff / real_cm

    results.append({
        "key": key,
        "direction": direction,
        "duration_ms": duration_ms,
        "real_cm": real_cm,
        "tick_diff": tick_diff,
        "cm_per_tick": round(cm_per_tick, 4),
        "ticks_per_cm": round(ticks_per_cm, 4),
    })

# === PRINT SUMMARY ===
print("\n=== Encoder Calibration Summary ===")
for r in results:
    print(f"Key '{r['key']}' | {r['direction']:^5} | {r['duration_ms']:>3} ms | "
          f"Δticks: {r['tick_diff']:>4} | {r['real_cm']:>5.1f} cm | "
          f"{r['cm_per_tick']:>7} cm/tick | {r['ticks_per_cm']:>7} ticks/cm")

# === STATS (excluding very small movements)
cm_per_tick_values = [r['cm_per_tick'] for r in results]
ticks_per_cm_values = [r['ticks_per_cm'] for r in results]

print("\n--- Summary Stats ---")
print(f"Average cm/tick   = {np.mean(cm_per_tick_values):.4f} ± {np.std(cm_per_tick_values):.4f}")
print(f"Average ticks/cm  = {np.mean(ticks_per_cm_values):.4f} ± {np.std(ticks_per_cm_values):.4f}")