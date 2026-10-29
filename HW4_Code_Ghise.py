import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score

CSV_PATH = "hw4-drug-screening-data.csv"
N_FOLDS = 5
RANDOM_STATE = 42

# Read CSV file
df = pd.read_csv(CSV_PATH)
print(df.shape)

# Separate the ids (in this case it's a pair id + drug name) and target labels
ID_COLS, TARGET = ["CELL_LINE_NAME", "DRUG_NAME"], "LN_IC50"
# Extract genes
genes = [c for c in df.columns if c not in ID_COLS + [TARGET]]

# Series of drug name and it's corresponding number of appearences
counts = df["DRUG_NAME"].value_counts()

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))
# Initialize each bar with drug's index and count
bars = ax.bar(counts.index, counts.values, color="#2a78d6", width=0.7)
ax.bar_label(bars, padding=3, fontsize=9, color="#333333")
ax.set_title(f"Samples per drug ({counts.size} drugs)", loc="center", fontsize=13)
ax.set_xlabel("Drug")
ax.set_ylabel("Number of samples")
ax.set_ylim(0, counts.max() * 1.2)
ax.tick_params(axis="x", labelrotation=45)
plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
# plt.show()


# Check for missing value in the data set
missing = df[genes].isna()
# Check if a gene has all missing values (in the specific column)
all_missing = missing.all()
# Filter which genes only have missing values
empty_genes = all_missing[all_missing].index.tolist()
print(f"Genes with all values missing: {len(empty_genes)}")
if (len(empty_genes) > 0):
 print(f"Empty genes: {empty_genes}")   

# Check for constant values in the data set
constant_genes = df[genes].columns[df[genes].nunique() <= 1].tolist()
print(f"Constant genes: {len(constant_genes)}")
if (len(constant_genes) > 0):
    print(f"Constant genes: {constant_genes}")
    
# Drop empty or constant genes
df = df.drop(columns=empty_genes+constant_genes)
# Update genes list
genes = [g for g in genes if g not in empty_genes + constant_genes]

print(df.shape)


def evaluate_model(y_test, y_pred):
    """Function to evaluate model

    Args:
        y_test (array): Array of actual values
        y_pred (array): Array of predicted values
        
    Returns:
        metrics (dict): Dictionary with MAE, MSE, RMSE and R2
    """
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

# Models to compare for every drug (add the other tree-based regressors here)
models = {
    "Decision Tree": lambda: DecisionTreeRegressor(
        max_depth=5,
        min_samples_leaf=10,
        random_state=RANDOM_STATE),
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=RANDOM_STATE),
    "Gradient Boosting Machine": lambda: GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=RANDOM_STATE), 
    "XGBoost": lambda: XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.5,     # each tree sees a random 50% of genes
        n_jobs=-1,
        random_state=RANDOM_STATE),
    "LightGBM": lambda: LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,            # LightGBM limits leaves instead of depth
        subsample=0.8,
        subsample_freq=1,         # needed for subsample to take effect
        colsample_bytree=0.5,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=-1),               # hide training warnings
    "CatBoost": lambda: CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=4,
        random_seed=RANDOM_STATE,
        verbose=0),              # don't print every training round
}

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
results = []

# Drug-specific regression: isolate each drug's data and cross-validate on it
for drug in sorted(df["DRUG_NAME"].unique()):
    # Select rows of the current drug
    drug_df = df[df["DRUG_NAME"] == drug]
    X = drug_df[genes]
    y = drug_df[TARGET]
    print(f"\n{drug} ({len(drug_df)} samples)")

    for model_name, make_model in models.items():
        scores = {
            "MAE": 0,
            "MSE": 0,
            "RMSE": 0,
            "R2": 0
        }

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # No scaling needed since we're using Decision Tree based algos
            model = make_model()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Evaluate model on the fold
            metrics = evaluate_model(y_test=y_test,
                                     y_pred=y_pred)

            # Aggregate model metrics
            for name, value in metrics.items():
                scores[name] += value

        # Calculate average metrics
        for name in scores:
            scores[name] /= N_FOLDS
        print(f"  {model_name}: " + "  ".join(f"{name}={value:.3f}" for name, value in scores.items()))

        results.append({"Drug": drug, "Model": model_name, **scores})

# One row per drug and model
results_df = pd.DataFrame(results)
print("\n", results_df.round(3).to_string(index=False))

# Best model per drug (lowest RMSE)
best_df = results_df.loc[results_df.groupby("Drug")["RMSE"].idxmin()]
print("\nBest model per drug (by RMSE):")
print(best_df.round(3).to_string(index=False))

# Save the full results table for the report
results_df.to_csv("drug_model_metrics.csv", index=False)


# One figure per drug: 4 bar plots (one per metric), 6 bars (one per model)
METRICS = ["MAE", "MSE", "RMSE", "R2"]
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

model_names = list(models.keys())
colors = dict(zip(model_names, MODEL_COLORS))

for drug in sorted(results_df["Drug"].unique()):
    # Metrics of every model for the current drug, in the same model order
    drug_results = results_df[results_df["Drug"] == drug].set_index("Model").loc[model_names]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, metric in zip(axes.flat, METRICS):
        values = drug_results[metric]
        bars = ax.bar(model_names, values, color=[colors[m] for m in model_names], width=0.7)
        labels = ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9, color="#333333")

        # Bold the best value (highest R2, lowest error)
        best = values.idxmax() if metric == "R2" else values.idxmin()
        labels[model_names.index(best)].set_fontweight("bold")

        lower = "higher is better" if metric == "R2" else "lower is better"
        ax.set_title(f"{metric} ({lower})", loc="left", fontsize=12)
        ax.set_ylabel(metric)
        # Leave room above (and below, for negative R2) for the value labels
        low, high = min(values.min(), 0), max(values.max(), 0)
        pad = (high - low) * 0.15
        ax.set_ylim(low - (pad if low < 0 else 0), high + pad)
        ax.axhline(0, color="#999999", linewidth=0.8)
        ax.tick_params(axis="x", labelrotation=30)
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
        ax.set_axisbelow(True)

    fig.suptitle(f"{drug}: {N_FOLDS}-fold CV metrics per model", fontsize=15)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{drug}.png"), dpi=150)

plt.show()
