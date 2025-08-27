"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Any


def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    function to perform one hot encoding on the input data

    Parameters:
        X : pd.DataFrame -> input data

    Returns:
        pd.DataFrame -> one-hot encoded data
    """

    df = X.copy()
    rows_count = len(df)

    # we again use the threshold that we defined earlier
    threshold = max(10, int(0.1 * rows_count))

    # prepare an empty dataframe
    parts = []

    # we start iterating through the columns and one-hot encode them
    for column in df.columns:
        row = df[column]

        # following the example from class, we consider boolean values as discrete -> one-hot encode them
        if pd.api.types.is_bool_dtype(row):
            dummies = pd.get_dummies(row.astype("category"), prefix=column, prefix_sep="__", dummy_na=True)
            parts.append(dummies)
            continue

        # floating point values are considered real
        if pd.api.types.is_float_dtype(row):
            parts.append(row.astype(float).to_frame(column))
            continue

        # we use our threshold value to decide if integer values are discrete or real
        if pd.api.types.is_integer_dtype(row):
            unique_count = row.dropna().nunique()
            if unique_count < threshold:
                dummies = pd.get_dummies(row.astype("category"), prefix=column, prefix_sep="__", dummy_na=True)
                parts.append(dummies)
            else:
                parts.append(row.astype(float).to_frame(column))

            continue

        # everything else is also discrete
        dummies = pd.get_dummies(row.astype("category"), prefix=column, prefix_sep="__", dummy_na=True)
        parts.append(dummies)

        final = pd.concat(parts, axis=1)

        return final.astype(float)



def numeric_is_real(y: pd.Series) -> bool:
    """
    helper function for check_ifreal to check if a numeric series is real or discrete

    Parameters:
        y : pd.Series -> numeric series

    Returns:
        bool -> True if series has real values, False otherwise
    """

    # if the values in the series are floating point, we consider it as real
    if pd.api.types.is_float_dtype(y):
        return True

    length = len(y)
    unique_length = y.nunique()

    # if the values are integers or any other numerical type, we check if most of the values are integers
    # if more than 10 values or more than 10% of values are numerical, return True
    threshold = max(10, int(0.1 * length))

    return unique_length > threshold


def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values

    Parameters:
        y : pd.Series -> the series to check for

    Returns:x
        bool -> True if the series has real (continuous) values, False otherwise
    """

    # we first remove the NaN values from the series
    y_new = y.dropna()

    # if the series is empty, we assume it to be discrete by default
    if y_new.empty:
        return False

    # if the series is of boolean type, we consider it as discrete
    # according to the tennis example given in class
    if pd.api.types.is_bool_dtype(y_new):
        return False

    # if the series is numeric, we use our helper function
    if pd.api.types.is_numeric_dtype(y_new):
        return numeric_is_real(y_new)

    # if the series is not numeric, we first try to see if it can be converted into a numeric series
    converted = pd.to_numeric(y_new, errors="coerce").dropna()

    # if 95% of the values get converted, we consider it to be numeric
    if len(converted) >= 0.95 * len(y_new):
        return numeric_is_real(converted)

    # we return false for all other cases, as the values would not be real
    return False


def entropy(Y: pd.Series) -> float:
    """
    function to calculate the entropy of a target variable Y

    formula: H(Y) = - Σ p_i * log2(p_i)

    Parameters:
        Y : pd.Series -> target variable

    Returns:
        float -> entropy value
    """

    # get unique class values and their counts
    values, counts = np.unique(Y, return_counts=True)

    # calculate probabilities for each class
    probabilities = counts / counts.sum()

    # apply entropy formula
    return -np.sum(probabilities * np.log2(probabilities))


def gini_index(Y: pd.Series) -> float:
    """
    function to calculate the gini index of a target variable Y

    formula: gini(Y) = 1 - Σ p_i^2

    Parameters:
        Y : pd.Series -> target variable

    Returns:
        float -> gini index value
    """

    # get unique class values and their counts
    values, counts = np.unique(Y, return_counts=True)

    # calculate probabilities for each class
    probabilities = counts / counts.sum()

    # apply gini formula
    return 1 - np.sum(probabilities ** 2)


def mse(Y: pd.Series) -> float:
    """
    function to calculate mean squared error

    Parameters:
        Y : pd.Series -> target variable

    Returns:
        float -> mean squared error
    """

    return np.mean((Y - np.mean(Y)) ** 2)


def information_gain(
    Y: pd.Series, attr: pd.Series, criterion: str = "entropy"
) -> float:
    """
    function to calculate the information gain of splitting dataset Y using attribute attr

    Parameters:
        Y : pd.Series -> target variable
        attr : pd.Series -> attribute used to split
        criterion : str -> "entropy", "gini", or "mse"

    Returns:
        float -> information gain value

    Raises:
        ValueError -> if criterion is not one of "entropy", "gain" or "mse"
    """

    # select impurity function based on criterion
    # base_impurity = impurity of parent set

    match criterion:
        case "entropy":
            base_impurity = entropy(Y)
            impurity_func = entropy
        case "gini":
            base_impurity = gini_index(Y)
            impurity_func = gini_index
        case "mse":
            mean_val = np.mean(Y)
            base_impurity = np.mean((Y - mean_val) ** 2)
            impurity_func = mse
        case _:
            raise ValueError("Criterion must be one of ['entropy', 'gini', 'mse']")


    weighted_impurity = 0.0

    # iterate over unique values of the attribute
    for val in np.unique(attr):
        # subset of Y where attr == val
        subset_Y = Y[attr == val]

        # proportion of subset
        weight = len(subset_Y) / len(Y)

        weighted_impurity += weight * impurity_func(subset_Y)

    # information gain = parent impurity - weighted impurity of children
    return base_impurity - weighted_impurity


# helper function to find impurity function based on given criterion
def find_impurity_func(criterion: str):
    """
    function to find impurity function according to given criterion

    Parameters:
        criterion : str -> "entropy", "gini" or "mse"

    Returns:
        function -> the impurity function

    Raises:
        ValueError -> if criterion is not one of "entropy", "gini" or "mse"
    """

    match criterion:
        case "entropy":
            return entropy
        case "gini":
            return gini_index
        case "mse":
            return  mse
        case _:
            return ValueError("Criterion must be one of ['entropy', 'gini', 'mse']")


# helper function for real input
def best_threshold_gain(x: pd.Series, y: pd.Series, criterion: str) -> Tuple[Optional[float], float]:
    """
    function to find the threshold on a real-valued feature, that maximizes information gain for the
    target variable under the given criterion

    Parameters:
        x : pd.Series -> real-valued feature
        y : pd.Series -> target variable
        criterion : str -> "entropy", "gini" or "mse"

    Returns:
        Tuple[float, float] -> (threshold that maximizes information gain, corresponding maximum information gain)
    """

    # for this function, we drop rows with NaN in both series
    valid_rows = (~x.isna()) & (~y.isna())
    x_new = x[valid_rows]
    y_new = y[valid_rows]

    # if there is one or less unique values, we cannot perform a split
    if x_new.size == 0 or np.unique(x_new).size <= 1:
        return None, 0.0

    # for mse criterion, we check if the target variable is numeric
    if (criterion == "mse") and (not np.issubdtype(y_new, np.number)):
        try:
            y_new = pd.to_numeric(y_new)
        except Exception as e:
            raise ValueError("For 'mse', target variable must be numeric") from e

    # first we sort by feature values
    order = np.argsort(x_new)
    x_new = x_new[order]
    y_new = y_new[order]

    impurity_func = find_impurity_func(criterion)
    impurity_val = impurity_func(pd.Series(y_new))

    # we initialize gain and threshold values
    best_gain = -np.inf
    best_threshold = None

    length = len(x_new)

    # we can consecutive unique values
    for i in range(1, length):
        # there will be no threshold between same values
        if x_new[i] == x_new[i - 1]:
            continue

        # find threshold = midpoint between consecutive distinct values
        threshold = 0.5 * (x_new[i] + x_new[i - 1])

        # we split the target variable according to the threshold
        left_y = y_new[:i]
        right_y = y_new[:i]

        # next we compute the weighted impurity of the children
        weight_left = i / length
        weight_right = 1 - weight_left
        child_impurity = (weight_left * impurity_func(pd.Series(left_y))) + (weight_right * impurity_func(pd.Series(right_y)))

        gain = impurity_val - child_impurity
        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    # if no valid threshold, we do not split
    if best_gain == -np.inf:
        return None, 0.0

    # if we get small negative gain value, we consider it as 0
    if 0 > best_gain > -1e-12:
        best_gain = 0.0

    return best_threshold, float(best_gain)


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series) -> Tuple[str, Optional[float], float]:
    """
    function to find the best attribute and threshold (for real values) to split about

    Parameters:
        X : pd.DataFrame -> feature dataframe
        y : pd.Series -> target variable
        criterion : str -> "entropy", "gini" or "mse"
        features : pd.Series -> list of attributes

    Returns:
        Tuple[str, float, float] -> (best feature, best threshold; None for discrete, best gain)
    """

    features = list(features)

    # initialize to track best possible values
    best_feature = None
    best_threshold = None
    best_gain = -np.inf\


    # iterate over all features
    for feature in features:
        feature_values = X[feature]

        # check if values are real or discrete
        is_real = check_ifreal(feature_values)

        if is_real:
            # find best threshold if real
            threshold, gain = best_threshold_gain(y, feature_values, criterion)
        else:
            # for discrete we use information gain
            valid_rows = (~feature_values.isna()) & (~y.isna())
            if valid_rows.sum() == 0:
                threshold, gain = None, 0.0
            else:
                gain = information_gain(y[valid_rows], feature_values[valid_rows], criterion)
                threshold = None

        # we keep track of maximum gain
        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold
            best_feature = feature

    # if none of the splits improves the impurity, return default values
    if best_feature is None:
        return features[0], None, 0.0

    # if gain is very small negative value consider as 0
    if 0 > best_gain > 1e-12:
        best_gain = 0.0

    return best_feature, best_threshold, float(best_gain)


# helper function to check if an object is an iterable except str
def is_iterable_not_str(obj: Any) -> bool:
    """
    function to find if an object is an iterable (except a string)

    Parameters:
        obj : Any -> object to check for

    Returns:
        bool -> True if an object is an iterable except a string, False otherwise
    """

    if isinstance(obj, (str, bytes)):
        return False

    try:
        iter(obj)
        return True
    except TypeError:
        return False


def split_data(X: pd.DataFrame, y: pd.Series, attribute: str, value: Any) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    function to split the data according to the given attribute

    Parameters:
        X : pd.DataFrame -> input data
        y : pd.Series -> target values
        attribute : str -> column in X to split from
        value : Any -> for real valued features = numeric threshold
                       for discrete features = a single category
    """

    # check if attribute is present in the dataframe
    if attribute not in X.columns:
        raise KeyError(f"Attribute '{attribute}' not present in the dataframe")

    feature_values = X[attribute]

    # check for discrete or real type
    is_real = check_ifreal(feature_values)

    # we do not consider rows where either one of the values is NaN
    valid_rows = feature_values.notna() & y.notna()

    # consider real and discrete values separately
    if is_real:
        if value is None:
            raise ValueError(
                f"Real-valued split on '{attribute}' requires a threshold value"
            )

        try:
            threshold = float(value)
        except Exception as e:
            raise ValueError(
                f"Threshold value for real feature '{attribute}' must be numeric"
            ) from e

        feature_values_new = feature_values[valid_rows]
        y_new = y[valid_rows]

        left = feature_values_new <= threshold
        right = feature_values_new > threshold

        X_left, y_left = X.loc[left.index[left]], y_new[left]
        X_right, y_right = X.loc[right.index[right]], y_new[right]

        return X_left, y_left, X_right, y_right

    else:
        if value is None:
            raise ValueError(f"Discrete-valued split on '{attribute}' requires a category value")

        feature_values_new = feature_values[valid_rows]
        y_new = y[valid_rows]

        # Binary split on a single category or a group of categories
        if is_iterable_not_str(value):
            left = feature_values_new.isin(set(value))
        else:
            left = feature_values_new == value

        right = (~left) & feature_values_new.notna()

        X_left, y_left = X.loc[left.index[left]], y_new[left]
        X_right, y_right = X.loc[right.index[right]], y_new[right]

        return X_left, y_left, X_right, y_right