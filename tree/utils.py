"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np


def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """

    pass


def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """

    pass


def entropy(Y: pd.Series) -> float:
    """
    function to calculate the entropy of a target variable Y.
    entropy measures the impurity or uncertainty in the dataset.

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
    function to calculate the gini index of a target variable Y.
    gini index measures impurity based on squared probabilities.

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


def information_gain(
    Y: pd.Series, attr: pd.Series, criterion: str = "entropy"
) -> float:
    """
    function to calculate the information gain of splitting dataset Y using attribute attr.

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
            impurity_func = lambda subset: np.mean((subset - np.mean(subset)) ** 2)
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

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).

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
