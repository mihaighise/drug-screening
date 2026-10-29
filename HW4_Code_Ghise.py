import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
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
        allow_writing_files=False,  # don't create the catboost_info folder
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


# One figure per drug: 4 groups on x (one per metric), 6 bars per group (one per model)
METRICS = ["MAE", "MSE", "RMSE", "R2"]
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

model_names = results_df["Model"].unique().tolist()
n_models = len(model_names)
bar_width = 0.8 / n_models          # the 6 bars of a group share 80% of the space
group_pos = np.arange(len(METRICS)) # x position of each metric group

for drug in sorted(results_df["Drug"].unique()):
    # Metrics of every model for the current drug, in the same model order
    drug_results = results_df[results_df["Drug"] == drug].set_index("Model").loc[model_names]

    # Best model of each metric (lowest error, highest R2)
    best = {m: drug_results[m].idxmax() if m == "R2" else drug_results[m].idxmin() for m in METRICS}

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (model_name, color) in enumerate(zip(model_names, MODEL_COLORS)):
        # Shift each model's bars so they sit side by side inside every group
        offset = (i - (n_models - 1) / 2) * bar_width
        bars = ax.bar(group_pos + offset, drug_results.loc[model_name, METRICS],
                      width=bar_width * 0.9, color=color, label=model_name)
        # Value on top of each bar, tilted so the 6 labels of a group don't overlap
        labels = ax.bar_label(bars, fmt="%.3f", padding=5, fontsize=8, rotation=45, color="#333333")
        # Put the best value of each metric inside a rectangle
        for label, metric in zip(labels, METRICS):
            if best[metric] == model_name:
                label.set_bbox(dict(boxstyle="square,pad=0.25", facecolor="white",
                                    edgecolor="#222222", linewidth=1))

    # Leave room above (and below, for negative R2) for the rotated labels
    low = min(drug_results[METRICS].min().min(), 0)
    high = drug_results[METRICS].max().max()
    pad = (high - low) * 0.15
    ax.set_ylim(low - (pad if low < 0 else 0), high + pad)

    ax.set_xticks(group_pos, METRICS)
    ax.set_title(f"Drug: {drug}", fontsize=15)
    ax.set_ylabel("Value")
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=n_models, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{drug}.png"), dpi=150)


# SHAP on the best regressor of each drug: top 10 most important genes
TOP_K = 10
SHAP_DIR = "shap_plots"
os.makedirs(SHAP_DIR, exist_ok=True)
top_features = []

for _, row in best_df.iterrows():
    drug, model_name = row["Drug"], row["Model"]
    drug_df = df[df["DRUG_NAME"] == drug]
    X = drug_df[genes]
    y = drug_df[TARGET]

    # Refit the best model on all of the drug's data
    model = models[model_name]()
    model.fit(X, y)

    # TreeExplainer computes exact SHAP values for tree-based models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    # Importance of a gene = mean |SHAP value| over all cell lines
    importance = pd.Series(np.abs(shap_values.values).mean(axis=0), index=genes)
    importance = importance.sort_values(ascending=False).head(TOP_K)
    print(f"\n{drug} ({model_name}) top {TOP_K} genes:")
    for rank, (gene, value) in enumerate(importance.items(), start=1):
        print(f"  {rank:2d}. {gene}: {value:.4f}")
        top_features.append({"Drug": drug, "Model": model_name, "Rank": rank,
                             "Gene": gene, "Mean |SHAP|": value})

    # Bar plot of only the top 10 genes (no "sum of other features" bar, which would dwarf them)
    top_idx = [genes.index(g) for g in importance.index]
    plt.figure()
    shap.plots.bar(shap_values[:, top_idx], max_display=TOP_K, show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    plt.title(f"Drug: {drug} ({model_name}): top {TOP_K} genes by mean |SHAP|", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(SHAP_DIR, f"{drug}_shap.png"), dpi=150)

# Save the top genes of every drug for the report
pd.DataFrame(top_features).to_csv("shap_top10_features.csv", index=False)

plt.show()
