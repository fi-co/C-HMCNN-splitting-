# Description: Script to run the main_split.py sequentially for 0->9 seeds and plot the results


import os

import time
import matplotlib.pyplot as plt
import subprocess
import pandas as pd

# Set parameters 
experiment_script = "main_split.py"
dataset = "cellcycle_FUN"
results_dir = "results"
output_file = f"{results_dir}/{dataset}_split_results.csv"
seeds = list(range(10))  
num_splits = 4
device = "0"  # If cpu

# Check that results directory exists
os.makedirs(results_dir, exist_ok=True)

# Step 1: execute command line
print(f"Run eperiments on {dataset} with {num_splits} splits and device {device}")

all_results = []

for i, seed in enumerate(seeds):
    print(f"\n [{i+1}/{len(seeds)}] Executing experiment with seed {seed}...")
    
    start_time = time.time()  # Start timer

    command = [
        "python", experiment_script,
        "--dataset", dataset,
        "--seed", str(seed),
        "--device", device,
        "--num_splits", str(num_splits)
    ]

    
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    execution_time = time.time() - start_time  # Calculate experiment time

    if process.returncode != 0:
        print(f" Error executing seed {seed}:")
        print(process.stderr)
        continue  # Bail out of the loop -> next seed
    
    
    print(f"[{i+1}/{len(seeds)}] Experiment with seed {seed} lasted {execution_time:.2f}s.")
    print(f"{len(seeds) - (i + 1)} experiments to go.")

    # Read results from CSV
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)

        if set(['seed', 'split', 'num_samples', 'average_precision']).issubset(df.columns):
            df["seed"] = seed  # Double check seed is present, or add it
            all_results.append(df)
        else:
            print(f"No matching columns found, seed {seed} skipped.")
    else:
        print(f"csv file not found with seed {seed}, skipped.")

# Concatenate all individual splits results, if any
if not all_results:
    print("No results to plot, bye-bye")
    exit(1)

results_df = pd.concat(all_results, ignore_index=True)




# Visualize the AP against the number of splits and seeds
plt.figure(figsize=(10, 6))

for seed in seeds:
    subset = results_df[results_df["seed"] == seed]
    plt.plot(subset["num_samples"], subset["average_precision"], marker="o", linestyle="-", alpha=0.6, label=f"Seed {seed}")

# compute the mean AP curve
mean_results = results_df.groupby("num_samples")["average_precision"].mean()
plt.plot(mean_results.index, mean_results.values, marker="o", linestyle="--", color="black", linewidth=2, label="Media")

plt.xlabel("Numero di campioni nel training set")
plt.ylabel("Average Precision (AP)")
plt.title(f"Andamento dell'AP per {dataset} con {num_splits} split")
plt.legend(loc="lower right", fontsize="small", ncol=2)
plt.grid(True)
plt.tight_layout()

# Salva il plot
plot_dir = f"{results_dir}/AP_vs_Splits.png"
plt.savefig(plot_dir)
plt.show()
print(f"Grafico salvato in {plot_dir}")