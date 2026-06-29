import pandas as pd
import numpy as np

# =====================================================
# IGD
# =====================================================

def igd(pareto_F, true_front):

    total = 0.0

    for p in true_front:

        d = np.min([
            np.sqrt(np.sum((p - q) ** 2))
            for q in pareto_F
        ])

        total += d

    return total / len(true_front)


# =====================================================
# GD
# =====================================================

def gd(pareto_F, true_front):

    total = 0.0

    for p in pareto_F:

        d = np.min([
            np.sqrt(np.sum((p - q) ** 2))
            for q in true_front
        ])

        total += d

    return total / len(pareto_F)


# =====================================================
# FOLDERS
# =====================================================

folders = ["2", "3", "4", "5"]

# =====================================================
# RUN ALL EXPERIMENTS
# =====================================================

for folder in folders:

    print("\n" + "=" * 60)
    print(f"RUN: {folder}")
    print("=" * 60)

    # -------------------------------------------------
    # Load all evaluations
    # -------------------------------------------------

    nsga_all = pd.read_csv(
        f"{folder}/results/nsga2_all_evals.csv"
    )

    grid_all = pd.read_csv(
        f"{folder}/results/grid_all_evals.csv"
    )

    rand_all = pd.read_csv(
        f"{folder}/results/rand_all_evals.csv"
    )

    # -------------------------------------------------
    # Global normalization
    # -------------------------------------------------

    all_data = pd.concat(
        [nsga_all, grid_all, rand_all],
        ignore_index=True
    )

    OBJECTIVES = [
        "error",
        "time_s",
        "co2_kg"
    ]

    mins = all_data[OBJECTIVES].min()
    maxs = all_data[OBJECTIVES].max()

    def normalize(df):

        F = (
            df[OBJECTIVES] - mins
        ) / (
            maxs - mins
        )

        F = np.clip(F, 0, 1)

        return F.values

    # -------------------------------------------------
    # Load Pareto fronts
    # -------------------------------------------------

    grid_pareto = pd.read_csv(
        f"{folder}/results/grid_pareto.csv"
    )

    nsga_pareto = pd.read_csv(
        f"{folder}/results/nsga2_pareto.csv"
    )

    rand_pareto = pd.read_csv(
        f"{folder}/results/rand_pareto.csv"
    )

    # -------------------------------------------------
    # Normalize fronts
    # -------------------------------------------------

    grid_F_norm = normalize(grid_pareto)

    nsga_F_norm = normalize(nsga_pareto)

    rand_F_norm = normalize(rand_pareto)

    # -------------------------------------------------
    # Compute metrics
    # -------------------------------------------------

    nsga_igd = igd(
        nsga_F_norm,
        grid_F_norm
    )

    nsga_gd = gd(
        nsga_F_norm,
        grid_F_norm
    )

    rand_igd = igd(
        rand_F_norm,
        grid_F_norm
    )

    rand_gd = gd(
        rand_F_norm,
        grid_F_norm
    )

    # -------------------------------------------------
    # Print
    # -------------------------------------------------

    print("\nNSGA-II")
    print(f"IGD = {nsga_igd:.6f}")
    print(f"GD  = {nsga_gd:.6f}")

    print("\nRandom Search")
    print(f"IGD = {rand_igd:.6f}")
    print(f"GD  = {rand_gd:.6f}")