from typing import Union
import pandas as pd
import numpy as np


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    function to calculate the accuracy

    Parameters:
        y_hat : pd.Series -> predicted labels
        y : pd.Series -> true labels

    Returns:
        float -> fraction of correct predictions
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """

    assert y_hat.size == y.size

    # we perform a position wise comparison as numpy arrays
    pred = y_hat.to_numpy()
    true = y.to_numpy()
    correct = (pred == true)

    return float(np.mean(correct))


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    function to calculate the precision

    Parameters:
        y_hat : pd.Series -> predicted labels
        y : pd.Series -> true labels
        cls : Union[int, str] -> class label for which precision has to be computed

    Returns:
        float -> precision for class 'cls'
    """

    # first we convert to numpy arrays
    pred = y_hat.to_numpy()
    true = y.to_numpy()

    # we first find predicted positives for the target class
    pred_pos = (pred == cls)

    # then we find the true positives
    tp = int(np.sum(pred_pos & (true == cls)))

    # we also need to find the False positives
    fp = int(np.sum(pred_pos & (true != cls)))

    # the total predicted positives
    denom = tp + fp
    if denom == 0:
        return 0.0

    # we can now return the precision
    return float(tp / denom)


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    function to calculate the recall

    Parameters:
        y_hat : pd.Series -> predicted labels
        y : pd.Series -> true labels
        cls : Union[int, str] -> class label for which recall has to be computed

    Returns:
        float -> recall for class 'cls'
    """

    # first we convert to numpy arrays
    pred = y_hat.to_numpy()
    true = y.to_numpy()

    # we find actual positives
    actual_pos = (true == cls)

    # then we find the true positives
    tp = int(np.sum((pred == cls) & actual_pos))

    # we also need to find the false positives
    fn = int(np.sum((pred != cls) & actual_pos))

    # we find the total actual positives
    denom = tp + fn
    if denom == 0:
        return 0.0

    return float(tp / denom)


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    function to calculate the root mean squared error

    Parameters:
        y_hat : pd.Series -> predicted numeric values
        y : pd.Series -> true numeric values

    Returns:
        float -> rmse value
    """
    # we try to coerce the series to numeric
    y_hat_num = pd.to_numeric(y_hat, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    assert not y_hat_num.isna().any(), "y_hat contains non numeric or NaN values"
    assert not y_num.isna().any(), "y contains non numeric or NaN values"

    # now we can compute the rmse
    predicted = y_hat_num.to_numpy(dtype=np.float64)
    true = y_num.to_numpy(dtype=np.float64)

    diff = predicted - true
    mse_val = np.mean(diff * diff)
    return float(np.sqrt(mse_val))


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    function to calculate the mean absolute error

    Parameters:
        y_hat : pd.Series -> predicted numeric values
        y : pd.Series -> true numeric values aligned

    Returns:
        float -> mean absolute error
    """

    # we first try to corece the series
    y_hat_num = pd.to_numeric(y_hat, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    assert not y_hat_num.isna().any(), "y_hat contains non-numeric or NaN values"
    assert not y_num.isna().any(), "y contains non-numeric or NaN values"

    # we can now compute the mae
    pred = y_hat_num.to_numpy(dtype=np.float64)
    true = y_num.to_numpy(dtype=np.float64)

    abs_err = np.abs(pred - true)
    return float(np.mean(abs_err))    

