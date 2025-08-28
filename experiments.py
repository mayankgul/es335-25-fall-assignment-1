import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from typing import Literal, Tuple, List, Dict, Any

np.random.seed(42)
num_average_time = 100  # Number of times to run each experiment to calculate the average values


# function to generate data
def make_data(
    N: int,
    M: int,
    input_kind: Literal["discrete", "real"],
    output_kind: Literal["discrete", "real"],
    noise: float = 0.1,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    function to create synthetic data
    """

    if input_kind == "discrete":
        X_np = np.random.randint(0, 2, size=(N, M)).astype(np.int64)
    else:
        X_np = np.random.randn(N, M).astype(np.float64)

    # linear weights to generate target
    w = np.random.randn(M)
    lin = X_np @ w

    if output_kind == "discrete":
        # we add the noise and threshold at 0
        y_np = (lin + noise * np.random.randn(N) > 0.0).astype(int)
    else:
        y_np = lin + noise * np.random.randn(N)

    X = pd.DataFrame(X_np, columns=[f"f{j}" for j in range(M)])
    y = pd.Series(y_np, name="y")
    return X, y


# helper function to time fit and predict
def time_fit_predict_once(
    X: pd.DataFrame,
    y: pd.Series,
    criterion_for_cls: Literal["information_gain", "gini_index"] = "information_gain",
    max_depth: int = 6,
) -> Tuple[float, float]:
    """
    function to time a single fit and average of predict() calls
    """

    # first we initialize the model
    model = DecisionTree(criterion=criterion_for_cls, max_depth=max_depth)

    # measure time to fit
    t0 = time.perf_counter()
    model.fit(X, y)
    t_fit = time.perf_counter() - t0

    # now we measure average time for predict
    t_pred_runs: List[float] = []
    for _ in range(num_average_time):
        t1 = time.perf_counter()
        _ = model.predict(X)
        t_pred_runs.append(time.perf_counter() - t1)
    t_pred = float(np.mean(t_pred_runs))

    return t_fit, t_pred


# helper function to time fit and predict for varying N
def run_sweep_over_N(
    Ns: List[int],
    M_fixed: int,
    input_kind: Literal["discrete", "real"],
    output_kind: Literal["discrete", "real"],
    max_depth: int = 6,
) -> Dict[str, Any]:
    """
    function to vary N for a fixed M and measure fit and predict times
    """

    fit_times, pred_times = [], []

    for N in Ns:
        X, y = make_data(N=N, M=M_fixed, input_kind=input_kind, output_kind=output_kind)
        t_fit, t_pred = time_fit_predict_once(X, y, max_depth=max_depth)
        fit_times.append(t_fit)
        pred_times.append(t_pred)

    return {"N": Ns, "M": M_fixed, "fit": np.array(fit_times), "pred": np.array(pred_times)}


# helper function to time fit and predict for varying N
def run_sweep_over_M(
    Ms: List[int],
    N_fixed: int,
    input_kind: Literal["discrete", "real"],
    output_kind: Literal["discrete", "real"],
    max_depth: int = 6,
) -> Dict[str, Any]:
    """
    function to vary M for a fixed N and measure fit and predict times
    """

    fit_times, pred_times = [], []
    for M in Ms:
        X, y = make_data(N=N_fixed, M=M, input_kind=input_kind, output_kind=output_kind)
        t_fit, t_pred = time_fit_predict_once(X, y, max_depth=max_depth)
        fit_times.append(t_fit)
        pred_times.append(t_pred)

    return {"M": Ms, "N": N_fixed, "fit": np.array(fit_times), "pred": np.array(pred_times)}


# helper functions for visual comparison
def scale_to_first(measured: np.ndarray, theory: np.ndarray) -> np.ndarray:
    """
    function to scale the theory curve to pass through the first measured point
    """

    eps = 1e-12
    s = (measured[0] / (theory[0] + eps)) if len(measured) and len(theory) else 1.0
    return theory * s


def theory_fit_vs_N(Ns: np.ndarray, M_fixed: int, input_kind: str) -> np.ndarray:
    """
    function to compute the training time curve vs N at fixed M
    """

    Ns = np.asarray(Ns, dtype=float)
    if input_kind == "real":
        return M_fixed * Ns * np.log2(Ns + 1.0)
    else:
        return M_fixed * Ns


def theory_fit_vs_M(Ms: np.ndarray, N_fixed: int, input_kind: str) -> np.ndarray:
    """
    function to compute the training time curve vs M at fixed N
    """

    Ms = np.asarray(Ms, dtype=float)
    if input_kind == "real":
        return Ms * (N_fixed * np.log2(N_fixed + 1.0))
    else:
        return Ms * N_fixed


def theory_pred_vs_N(Ns: np.ndarray, depth: int = 6) -> np.ndarray:
    """
    function to compute the prediction time curve vs N at fixed depth
    """
    Ns = np.asarray(Ns, dtype=float)
    return Ns * max(depth, 1)


def theory_pred_vs_M(Ms: np.ndarray) -> np.ndarray:
    """
    function to compute the prediction time curve vs M at fixed N
    """

    Ms = np.asarray(Ms, dtype=float)
    return np.ones_like(Ms)


# now we can define functions to plot the graphs

def plot_sweep_vs_N(result: Dict[str, Any], input_kind: str, output_kind: str, depth: int = 6) -> None:
    Ns = np.array(result["N"], dtype=float)
    fit_meas = result["fit"]
    pred_meas = result["pred"]
    M_fixed = result["M"]

    # theory shapes
    fit_theory = scale_to_first(fit_meas, theory_fit_vs_N(Ns, M_fixed, input_kind))
    pred_theory = scale_to_first(pred_meas, theory_pred_vs_N(Ns, depth))

    plt.figure(figsize=(8, 5))
    plt.plot(Ns, fit_meas, "o-", label="fit (measured)")
    plt.plot(Ns, fit_theory, "--", label="fit (theory shape, scaled)")
    plt.plot(Ns, pred_meas, "s-", label="predict (measured)")
    plt.plot(Ns, pred_theory, "--", label="predict (theory shape, scaled)")
    plt.xlabel("N (samples)")
    plt.ylabel("time (seconds)")
    plt.title(f"Runtime vs N | inputs={input_kind}, output={output_kind}, M={M_fixed}")
    plt.legend()
    plt.tight_layout()


def plot_sweep_vs_M(result: Dict[str, Any], input_kind: str, output_kind: str, depth: int = 6) -> None:
    Ms = np.array(result["M"], dtype=float)
    fit_meas = result["fit"]
    pred_meas = result["pred"]
    N_fixed = result["N"]

    fit_theory = scale_to_first(fit_meas, theory_fit_vs_M(Ms, N_fixed, input_kind))
    pred_theory = scale_to_first(pred_meas, theory_pred_vs_M(Ms))

    plt.figure(figsize=(8, 5))
    plt.plot(Ms, fit_meas, "o-", label="fit (measured)")
    plt.plot(Ms, fit_theory, "--", label="fit (theory shape, scaled)")
    plt.plot(Ms, pred_meas, "s-", label="predict (measured)")
    plt.plot(Ms, pred_theory, "--", label="predict (theory-shape, scaled)")
    plt.xlabel("M (features)")
    plt.ylabel("time (seconds)")
    plt.title(f"Runtime vs M | inputs={input_kind}, output={output_kind}, N={N_fixed}")
    plt.legend()
    plt.tight_layout()


# finally we define a function to perform the experiments
def run_all_experiments() -> None:
    """
    function to run for the 4 cases with both discrete and real inputs and outputs
    """

    # we need to vary both the features and samples before plotting
    Ns = [500, 1000, 2000, 4000]
    Ms = [5, 10, 20, 40, 80]

    M_fixed_for_N = 20
    N_fixed_for_M = 2000

    depth = 6

    cases = [
        ("discrete", "discrete"),
        ("real", "real"),
        ("real", "discrete"),
        ("discrete", "real"),
    ]

    for input_kind, output_kind in cases:
        # sweep vs N
        res_N = run_sweep_over_N(Ns, M_fixed_for_N, input_kind, output_kind, max_depth=depth)
        plot_sweep_vs_N(res_N, input_kind, output_kind, depth=depth)

        # sweep vs M
        res_M = run_sweep_over_M(Ms, N_fixed_for_M, input_kind, output_kind, max_depth=depth)
        plot_sweep_vs_M(res_M, input_kind, output_kind, depth=depth)

    plt.show()


if __name__ == "__main__":
    run_all_experiments()