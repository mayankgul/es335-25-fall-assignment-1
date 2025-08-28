import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold

np.random.seed(42)

# Code given in the question
X_np, y_np = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np)
plt.title("Toy classification dataset (colored by class)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.tight_layout()
plt.show()

# Write the code for Q2 a) and b) below. Show your results.


# we first onvert to pandas for our DecisionTree
X = pd.DataFrame(X_np, columns=["x1", "x2"])
y = pd.Series(y_np, name="target")

# Q2 (a)

# we make a 70-30 split with reproducible shuffle
n = len(y)
perm = np.random.permutation(n)
split = int(0.7 * n)
train_idx, test_idx = perm[:split], perm[split:]

X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

# first we train th decision tree
tree = DecisionTree(criterion="information_gain", max_depth=5)
tree.fit(X_train, y_train)

# then we need to predict on the test set
y_pred = tree.predict(X_test)

# we can now compute metrics on the test set
acc = accuracy(y_pred, y_test)

# we find the distinct classes in the test set
classes = sorted(pd.unique(y_test))
per_class = []
for cls in classes:
    prec = precision(y_pred, y_test, cls=cls)
    rec = recall(y_pred, y_test, cls=cls)
    per_class.append((cls, prec, rec))

print("Q2 (a): 70-30 Train/Test Evaluation")
print(f"Test Accuracy: {acc:.4f}")
for cls, prec, rec in per_class:
    print(f"Class {cls}: Precision={prec:.4f}, Recall={rec:.4f}")
print()

# Q2 (b)

# first we need to perform 5 fold outer CV for evaluation
# then inside each outer train fold we run an inner 5 fold CV
outer_k = 5
inner_k = 5
depth_grid = list(range(1, 11))

outer_cv = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=42)

# best depth picked on each outer fold
selected_depths = []
# accuracy on the outer test fold
outer_fold_scores = []

fold_id = 0
for outer_train_idx, outer_test_idx in outer_cv.split(X, y):
    fold_id += 1
    X_tr, y_tr = X.iloc[outer_train_idx], y.iloc[outer_train_idx]
    X_te, y_te = X.iloc[outer_test_idx], y.iloc[outer_test_idx]

    # inner CV to select best depth
    inner_cv = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=42 + fold_id)
    depth_to_scores = {d: [] for d in depth_grid}

    for inner_train_idx, inner_val_idx in inner_cv.split(X_tr, y_tr):
        X_in_tr, y_in_tr = X_tr.iloc[inner_train_idx], y_tr.iloc[inner_train_idx]
        X_in_val, y_in_val = X_tr.iloc[inner_val_idx], y_tr.iloc[inner_val_idx]

        for d in depth_grid:
            model = DecisionTree(criterion="information_gain", max_depth=d)
            model.fit(X_in_tr, y_in_tr)
            y_val_pred = model.predict(X_in_val)
            depth_to_scores[d].append(accuracy(y_val_pred, y_in_val))

    # we find average inner value accuracy per depth and choose argmax
    avg_scores = {d: float(np.mean(scores)) if len(scores) > 0 else 0.0
                  for d, scores in depth_to_scores.items()}
    best_depth = max(avg_scores, key=avg_scores.get)

    selected_depths.append(best_depth)

    # we retrain on full outer train with chosen depth
    final_model = DecisionTree(criterion="information_gain", max_depth=best_depth)
    final_model.fit(X_tr, y_tr)
    y_outer_pred = final_model.predict(X_te)
    outer_acc = accuracy(y_outer_pred, y_te)
    outer_fold_scores.append(outer_acc)

    print(f"Outer fold {fold_id} best_depth={best_depth} | outer fold accuracy={outer_acc:.4f}")

print("\n Q2 (b): Nested 5 fold CV")
print(f"Chosen depths per outer fold: {selected_depths}")

# optimum can be the most frequently selected depth, that is the mode
vals, counts = np.unique(selected_depths, return_counts=True)
opt_depth_by_mode = int(vals[np.argmax(counts)])
print(f"Depth selected most often (mode): {opt_depth_by_mode}")

print(f"Average selected depth: {np.mean(selected_depths):.2f}")
print(f"Mean outer 5 fold accuracy: {np.mean(outer_fold_scores):.4f} ± {np.std(outer_fold_scores):.4f}")