import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn

# cleaning the data

# we drop the free text 'car name' for regression
df = data.copy()
df = df.drop(columns=["car name"])

# 'horsepower' has ? placeholders so we need to coerce to numeric and not consider NaN values
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
df = df.dropna().reset_index(drop=True)

# now we split features
y = df["mpg"]
X = df.drop(columns=["mpg"])

# we need to train/test with 80-20 split
n = len(y)
perm = np.random.permutation(n)
split = int(0.8 * n)
train_idx, test_idx = perm[:split], perm[split:]

X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

# Q3 (a)
# we first check for our implementation of decision tree

my_tree = DecisionTree(criterion="information_gain", max_depth=6)
my_tree.fit(X_train, y_train)
y_pred_custom = my_tree.predict(X_test)

rmse_custom = rmse(y_pred_custom, y_test)
mae_custom = mae(y_pred_custom, y_test)

print("Auto MPG: Our DecisionTree")
print(f"Test RMSE: {rmse_custom:.4f}")
print(f"Test MAE : {mae_custom:.4f}")

print()


# Q3 (b)
# we can now use sklearn implementation of decision tree
sk_model = DecisionTreeRegressor(max_depth=6, random_state=42)
sk_model.fit(X_train, y_train)
y_pred_sklearn = pd.Series(sk_model.predict(X_test), index=y_test.index)

rmse_sklearn = rmse(y_pred_sklearn, y_test)
mae_sklearn = mae(y_pred_sklearn, y_test)

print("Auto MPG: sklearn decision tree")
print(f"Test RMSE: {rmse_sklearn:.4f}")
print(f"Test MAE : {mae_sklearn:.4f}")

print()

# now we can compare the results with ours
print("Comparison")
print(f"RMSE -> our decision tree: {rmse_custom:.4f} | sklearn: {rmse_sklearn:.4f}")
print(f"MAE  -> our decision tree: {mae_custom:.4f} | sklearn: {mae_sklearn:.4f}")

# we can also plot the results
plt.figure()
plt.scatter(y_test, y_pred_custom, alpha=0.6, label="Our Decision Tree", marker="o")
plt.scatter(y_test, y_pred_sklearn, alpha=0.6, label="sklearn", marker="x")
mn, mx = y_test.min(), y_test.max()
plt.plot([mn, mx], [mn, mx])  # y=x reference
plt.xlabel("True MPG")
plt.ylabel("Predicted MPG")
plt.title("Auto MPG: Predictions vs Truth (Test Set)")
plt.legend()
plt.tight_layout()
plt.show()