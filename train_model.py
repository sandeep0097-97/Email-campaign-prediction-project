# %% [markdown]
# # InspireDirect: Email Campaign Effectiveness Prediction
#
# **Goal:** predict what a recipient does with a marketing email:
# `0 = Ignored`, `1 = Read`, `2 = Acknowledged`.
#
# Steps in this script:
# 1. Load and understand the data
# 2. Exploratory data analysis (EDA)
# 3. Prepare features (missing values, encoding)
# 4. Train and compare several models
# 5. Tune the best model and evaluate it
# 6. Explain which features matter
# 7. Save the model for the web app (api/model.joblib)
#
# Run from the project root:  `python src/train_model.py`

# %%
import json
import os
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")  # save charts to files (works without a screen)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, recall_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Make "api/features.py" importable, so training and the web app share one feature function
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "api"))
from features import CLASS_NAMES, FEATURE_NAMES, prepare_features  # noqa: E402

DATA_PATH = os.path.join(ROOT, "data", "data_email_campaign.csv")
FIG_DIR = os.path.join(ROOT, "reports", "figures")
MODEL_PATH = os.path.join(ROOT, "api", "model.joblib")
METRICS_PATH = os.path.join(ROOT, "reports", "metrics.json")
os.makedirs(FIG_DIR, exist_ok=True)

RANDOM_STATE = 42
PALETTE = ["#9AA5B1", "#2F7F6F", "#E0A030"]  # ignored / read / acknowledged
sns.set_theme(style="whitegrid", font_scale=1.0)


def save_fig(name):
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved chart -> reports/figures/{name}")


# %% [markdown]
# ## 1. Load and understand the data

# %%
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print(df.head())
print("\nMissing values per column:")
print(df.isna().sum()[df.isna().sum() > 0])

target_share = df["Email_Status"].value_counts(normalize=True).sort_index()
print("\nTarget distribution (share of emails):")
for cls, share in target_share.items():
    print(f"  {cls} = {CLASS_NAMES[cls]:<13} {share:6.1%}")
# Key insight: the classes are IMBALANCED (~80% ignored), so accuracy alone is misleading.
# We will judge models mainly by MACRO F1 (average F1 across the three classes).

# %% [markdown]
# ## 2. Exploratory data analysis

# %%
# 2a. Target distribution
plt.figure(figsize=(6, 4))
counts = df["Email_Status"].value_counts().sort_index()
plt.bar([CLASS_NAMES[i] for i in counts.index], counts.values, color=PALETTE)
for i, v in enumerate(counts.values):
    plt.text(i, v, f"{v:,}", ha="center", va="bottom")
plt.title("How recipients responded")
plt.ylabel("Number of emails")
save_fig("01_target_distribution.png")

# 2b. Missing values
missing = df.isna().mean().sort_values(ascending=False)
missing = missing[missing > 0] * 100
plt.figure(figsize=(6, 3.5))
plt.barh(missing.index, missing.values, color="#2F7F6F")
plt.xlabel("% missing")
plt.title("Missing values")
save_fig("02_missing_values.png")

# 2c. Campaign type vs response (the strongest single signal)
ct = pd.crosstab(df["Email_Campaign_Type"], df["Email_Status"], normalize="index") * 100
ct.columns = [CLASS_NAMES[c] for c in ct.columns]
ax = ct.plot(kind="bar", stacked=True, color=PALETTE, figsize=(7.5, 4))
ax.set_ylabel("% of emails")
ax.set_xlabel("Email campaign type")
ax.set_title("Response by campaign type")
plt.xticks(rotation=0)
plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9, frameon=False)
save_fig("03_campaign_type_vs_status.png")
print("\nResponse % by campaign type:\n", ct.round(1))

# 2d. Numeric features vs response
num_cols = ["Subject_Hotness_Score", "Total_Past_Communications", "Word_Count",
            "Total_Links", "Total_Images"]
fig, axes = plt.subplots(1, len(num_cols), figsize=(18, 4))
for ax, col in zip(axes, num_cols):
    sns.boxplot(data=df, x="Email_Status", y=col, hue="Email_Status",
                palette=PALETTE, legend=False, ax=ax, showfliers=False)
    ax.set_xticks([0, 1, 2], ["Ign.", "Read", "Ack."])
    ax.set_xlabel("")
    ax.set_title(col.replace("_", " "), fontsize=10)
save_fig("04_numeric_vs_status.png")
print("\nAverage of numeric features by response:")
print(df.groupby("Email_Status")[num_cols].mean().round(2))

# 2e. Correlation heatmap
plt.figure(figsize=(8, 6.5))
corr = df.drop(columns=["Email_ID", "Customer_Location"]).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="BrBG", center=0, annot_kws={"size": 7})
plt.title("Correlation between features")
save_fig("05_correlation_heatmap.png")

# %% [markdown]
# ## 3. Prepare features
# * Drop `Email_ID` (just an identifier, carries no signal).
# * One-hot encode `Customer_Location`; missing locations become their own "Unknown" group.
# * Missing numbers are filled with the median inside the model pipeline (SimpleImputer),
#   so the web app gets the same treatment automatically.

# %%
X = prepare_features(df.drop(columns=["Email_ID", "Email_Status"]).to_dict("records"))
y = df["Email_Status"].to_numpy()
print("Feature matrix:", X.shape, "| features:", FEATURE_NAMES)

# Stratified split keeps the same class mix in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print(f"Train: {len(y_train):,} rows | Test: {len(y_test):,} rows")

# %% [markdown]
# ## 4. Train and compare models
# `class_weight="balanced"` tells each model to pay more attention to the rare classes
# (Read, Acknowledged). Without it, a model can score 80% accuracy by predicting
# "Ignored" for everything, which is useless for the business.

# %%
models = {
    "Logistic Regression": Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ]),
    "Decision Tree": Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(max_depth=8, class_weight="balanced",
                                         random_state=RANDOM_STATE)),
    ]),
    "Random Forest": Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                         class_weight="balanced", n_jobs=-1,
                                         random_state=RANDOM_STATE)),
    ]),
    "Gradient Boosting": Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(class_weight="balanced",
                                                 random_state=RANDOM_STATE)),
    ]),
}


def evaluate(model, X_eval, y_eval):
    pred = model.predict(X_eval)
    recalls = recall_score(y_eval, pred, average=None)
    return {
        "accuracy": round(float(accuracy_score(y_eval, pred)), 4),
        "macro_f1": round(float(f1_score(y_eval, pred, average="macro")), 4),
        "recall_ignored": round(float(recalls[0]), 4),
        "recall_read": round(float(recalls[1]), 4),
        "recall_acknowledged": round(float(recalls[2]), 4),
    }


results = {}
for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    results[name] = evaluate(model, X_test, y_test)
    print(f"{name:<20} {results[name]}  ({time.time() - start:.1f}s)")

results_df = pd.DataFrame(results).T.sort_values("macro_f1", ascending=False)
print("\nModel comparison (sorted by macro F1):\n", results_df)

plt.figure(figsize=(7, 4))
plt.barh(results_df.index[::-1], results_df["macro_f1"][::-1], color="#2F7F6F")
for i, v in enumerate(results_df["macro_f1"][::-1]):
    plt.text(v, i, f" {v:.3f}", va="center")
plt.xlabel("Macro F1 (higher is better)")
plt.title("Model comparison on the test set")
plt.xlim(0, max(results_df["macro_f1"]) * 1.2)
save_fig("06_model_comparison.png")

# %% [markdown]
# ## 5. Tune the two strongest model families
# RandomizedSearchCV tries random combinations of settings with 3-fold cross-validation,
# using only the TRAINING data. We tune Random Forest and Gradient Boosting, then keep
# whichever has the better cross-validated macro F1. The test set stays untouched until
# the final check, so the final score is an honest estimate.

# %%
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
search_spaces = {
    "Random Forest": {
        "model__n_estimators": [150, 200, 300],
        "model__max_depth": [10, 12, 14],          # capped so the saved model stays small
        "model__min_samples_leaf": [5, 10, 20],
        "model__max_features": ["sqrt", 0.5],
    },
    "Gradient Boosting": {
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_iter": [200, 300, 400],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__min_samples_leaf": [20, 50, 100],
        "model__l2_regularization": [0.0, 0.5, 1.0],
    },
}
searches = {}
for name, space in search_spaces.items():
    start = time.time()
    searches[name] = RandomizedSearchCV(
        models[name], space, n_iter=8, scoring="f1_macro", cv=cv,
        random_state=RANDOM_STATE, n_jobs=-1).fit(X_train, y_train)
    print(f"{name:<18} best CV macro F1 = {searches[name].best_score_:.4f} "
          f"({time.time() - start:.0f}s)\n   params: {searches[name].best_params_}")

best_name = max(searches, key=lambda n: searches[n].best_score_)
search = searches[best_name]
best_model = search.best_estimator_
final_metrics = evaluate(best_model, X_test, y_test)
print(f"\nWinner: {best_name}")
print("Tuned model on the test set:", final_metrics)

# %% [markdown]
# ## 6. Evaluate in detail

# %%
pred = best_model.predict(X_test)
labels = [CLASS_NAMES[i] for i in range(3)]
print(classification_report(y_test, pred, target_names=labels))

cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt=",d", cmap="BuGn", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion matrix (test set)")
save_fig("07_confusion_matrix.png")

# %% [markdown]
# ## 7. Which features matter? (permutation importance)
# Shuffle one feature at a time and measure how much macro F1 drops.
# A big drop means the model relies on that feature.

# %%
perm = permutation_importance(best_model, X_test, y_test, scoring="f1_macro",
                              n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
importance = (pd.Series(perm.importances_mean, index=FEATURE_NAMES)
              .sort_values(ascending=False))
print("Feature importance (drop in macro F1 when shuffled):\n", importance.round(4))

top = importance.head(10)[::-1]
plt.figure(figsize=(7, 4.5))
plt.barh(top.index, top.values, color="#2F7F6F")
plt.xlabel("Drop in macro F1 when shuffled")
plt.title("Top 10 features")
save_fig("08_feature_importance.png")

# %% [markdown]
# ## 8. Save the model and metrics for the web app

# %%
joblib.dump(best_model, MODEL_PATH, compress=3)
size_mb = os.path.getsize(MODEL_PATH) / 1e6
print(f"Model saved -> api/model.joblib ({size_mb:.2f} MB)")

report = {
    "dataset_rows": int(len(df)),
    "class_share": {CLASS_NAMES[k]: round(float(v), 4) for k, v in target_share.items()},
    "model_comparison": results,
    "final_model": best_name,
    "final_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
    "final_cv_macro_f1": round(float(search.best_score_), 4),
    "final_metrics": final_metrics,
    "confusion_matrix": cm.tolist(),
    "feature_importance": {k: round(float(v), 4) for k, v in importance.items()},
}
with open(METRICS_PATH, "w") as f:
    json.dump(report, f, indent=2)
print("Metrics saved -> reports/metrics.json")
print("\nDone. Next: run the web app locally with  python api/index.py")
