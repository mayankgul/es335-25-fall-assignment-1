"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np


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
    return 1 - np.sum(probabilities**2)


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


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to whether the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or variance based on the type of output) or minimum gini index (discrete output).

    pass


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.

    pass
