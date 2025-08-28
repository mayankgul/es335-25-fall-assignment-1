"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)


# helper class to handle node structure
@dataclass
class TreeNode:
    is_leaf: bool
    prediction: Optional[Any] = None
    feature: Optional[str] = None
    threshold: Optional[float] = None
    categories_left: Optional[set] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    depth: int = 0
    n_samples: int = 0
    impurity: float = 0.0


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion: Literal["information_gain", "gini_index"], max_depth: int = 5):
        """
        initialize a decision tree

        Parameters:
            criterion : str -> "information_gain" or "gini_index"
            max_depth : int -> maximum depth of decision tree
        """

        self.criterion = criterion
        self.max_depth = max_depth

        # placeholder values
        self.root: Optional[TreeNode] = None
        self.is_fitted: bool = False
        self.n_features: Optional[int] = None
        self.feature_names: Optional[list[str]] = None
        self.classes: Optional[np.ndarray] = None
        self.task: Optional[str] = None

    # helper function to predict if node is a leaf node
    def leaf_prediction(self, y_sub: pd.Series):
        """
        function to prediction at a leaf node
        """

        if self.task == "regression":
            # mean minimizes squared error
            return float(np.mean(y_sub)) if len(y_sub) else np.nan

        # classification -> majority class (mode), use index in case of a tie
        counts = y_sub.value_counts(dropna=True)
        return counts.idxmax() if len(counts) else np.nan

    # helper function to build the tree
    def build_node(self, X_sub: pd.DataFrame, y_sub: pd.Series, depth: int, node_impurity_function, criterion_name) -> TreeNode:
        """
        function to grow the tree and return the root node of this subtree
        """

        n = len(y_sub)

        # compute impurity
        try:
            impurity = float(node_impurity_function(y_sub)) if n > 0 else 0.0
        except Exception:
            impurity = 0.0

        # create a node placeholder
        node = TreeNode(
            is_leaf=False,
            prediction=None,
            feature=None,
            threshold=None,
            categories_left=None,
            left=None,
            right=None,
            depth=depth,
            n_samples=n,
            impurity=impurity,
        )

        # we define the stopping conditions

        # if the node is empty
        if n == 0:
            node.is_leaf = True
            node.prediction = np.nan
            return node

        # if we have reached the maximum depth
        if depth >= int(self.max_depth):
            node.is_leaf = True
            node.prediction = self.leaf_prediction(y_sub)
            return node

        # if we get a pure node or near-constant value
        if self.task == "classification":
            if pd.unique(y_sub).size <= 1:
                node.is_leaf = True
                node.prediction = self.leaf_prediction(y_sub)
                return node
        else:
            # if variance is 0 or we have only one or less values
            if n <= 1 or float(np.var(y_sub)) <= 1e-12:
                node.is_leaf = True
                node.prediction = self.leaf_prediction(y_sub)
                return node

        # now we find the best split
        feat_name, threshold, gain = opt_split_attribute(
            X_sub, y_sub, criterion_name, X_sub.columns
        )

        # if we do not get a useful split, consider as leaf node
        if gain <= 0.0 or feat_name is None:
            node.is_leaf = True
            node.prediction = self.leaf_prediction(y_sub)
            return node

        # now we split the data recursively
        try:
            X_left, y_left, X_right, y_right = split_data(
                X_sub, y_sub, attribute=feat_name, value=threshold
            )
        except Exception:
            # if the split fails, consider as a leaf node
            node.is_leaf = True
            node.prediction = self.leaf_prediction(y_sub)
            return node

        # if one side is empty we do not need to split
        if len(y_left) == 0 or len(y_right) == 0:
            node.is_leaf = True
            node.prediction = self.leaf_prediction(y_sub)
            return node

        # we fill split info for the node
        node.feature = feat_name
        node.threshold = float(threshold) if threshold is not None else None
        node.left = self.build_node(X_left, y_left, depth + 1)
        node.right = self.build_node(X_right, y_right, depth + 1)
        return node


    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        function to train and construct the decision tree

        Parameters:
            X : pd.DataFrame -> input data
            y : pd.Series -> target variable
        """

        # keep a copy aligned to the dataframe
        y = y.loc[X.index]

        # check if the values are real or discrete
        is_y_real = check_ifreal(y)
        self.task = "regression" if is_y_real else "classification"

        if self.task == "classification":
            criterion_name = "entropy" if self.criterion == "information_gain" else "gini"
            node_impurity_function = entropy if self.criterion == "information_gain" else gini_index
        else:
            criterion_name = "mse"
            node_impurity_function = mse

        # perform one-hot encoding on the dataframe as required
        X_enc = one_hot_encoding(X)

        self.feature_names = list(X_enc.columns)
        self.n_features = len(self.feature_names)

        # drop rows with missing y
        valid_rows = y.notna()
        X_enc = X_enc.loc[valid_rows]
        y = y.loc[valid_rows]

        # store class labels for classification
        if self.task == "classification":
            self.classes = np.array(pd.unique(y))

        # we can now build the tree
        self.root = self.build_node(X_enc, y, depth=0, node_impurity_function=node_impurity_function, criterion_name=criterion_name)
        self.is_fitted = True


    # helper function to traverse a row of the learned tree
    def predict_row(self, row: pd.Series):
        """
        function to compute the prediction for a single input row

        Parameters:
            row : pd.Series -> one-hot encoded feature vector
        """
        node = self.root

        # traverse the tree until we hit a leaf
        while node is not None and not node.is_leaf:
            feature = node.feature
            threshold = node.threshold

            # if split info is missing we exit the loop
            if feature is None or threshold is None:
                break

            # we then find the value for this row's feature
            value = row.get(feature, 0.0)

            # if the value is NaN, we route to the child with more samples
            if pd.isna(value):
                if node.left is None and node.right is None:
                    break
                if node.left is None:
                    node = node.right
                    continue
                if node.right is None:
                    node = node.left
                    continue
                node = node.left if node.left.n_samples >= node.right.n_samples else node.right
                continue

            # if the features are real, we do a threshold split
            if value <= threshold:
                node = node.left
            else:
                node = node.right

        # if we reach a leaf node, return the stored prediction
        if node is not None and node.is_leaf:
            return node.prediction

        # otherwise we return NaN
        return np.nan if self.task == "regression" else pd.NA

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        function to run the decision tree on test inputs

        Parameters:
            X : pd.DataFrame -> the test input dataframe

        Returns:
            pd.Series -> series containing one prediction per input row
        """

        # We must replicate training-time preprocessing: one-hot for discrete,
        # keep real-valued columns, then cast to float.
        X_enc = one_hot_encoding(X)

        # first we align to the training feature set and fill any missing values
        X_enc = X_enc.reindex(columns=self.feature_names, fill_value=0.0)

        # now we traverse the tree row wise and return a series
        prediction = [self.predict_row(X_enc.loc[idx]) for idx in X_enc.index]
        return pd.Series(prediction, index=X.index, name="prediction")


    # function to pretty print numbers
    def fmt_num(self, x: float) -> str:
        """
        function to pretty print numbers with a few significant digits
        """
        try:
            return f"{float(x):.6g}"
        except Exception:
            return str(x)

    # helper function to render the node split
    def condition_str(self, node: TreeNode) -> str:
        """
        function to render the node's split in the format as required
        """

        if node.feature is None or node.threshold is None:
            return "?(unknown split)"

        threshold = float(node.threshold)

        if abs(threshold - 0.5) <= 1e-9:
            return f"?({node.feature} == 1)"

        return f"?({node.feature} > {self.fmt_num(threshold)})"

    # function to render a leaf prediction
    def leaf_str(self, node: TreeNode) -> str:
        """
        function to render a leaf prediction
        """

        pred = node.prediction
        if self.task == "regression":
            return self.fmt_num(pred)
        # classification
        return f"Class {pred}"

    # helper function to render the lines to be displayed
    def render(self, node: TreeNode, indent: int, preface: str | None = None) -> list[str]:
        padding = "    " * indent
        lines: list[str] = []

        if node.is_leaf:
            leaf = self.leaf_str(node)

            if preface:
                lines.append(f"{padding}{preface} {leaf}")
            else:
                lines.append(f"{padding}{leaf}")
            return lines

        # if it is an internal node, we write the condition in one line
        cond = self.condition_str(node)
        if preface:
            lines.append(f"{padding}{preface} {cond}")
        else:
            lines.append(f"{padding}{cond}")

        # for a right child
        if node.right is not None:
            lines.extend(self.render(node.right, indent + 1, "Y:"))
        else:
            lines.append(f"{'    ' * (indent + 1)}Y: (empty)")

        # for left child
        if node.left is not None:
            lines.extend(self.render(node.left, indent + 1, "N:"))
        else:
            lines.append(f"{'    ' * (indent + 1)}N: (empty)")

        return lines


    def plot(self) -> None:
        """
        function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """

        # generate lines using helper functions and print the tree
        text_lines = self.render(self.root, indent=0, preface=None)
        print("\n".join(text_lines))
