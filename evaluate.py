# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Evaluation Functions for Regression Models
#

# +
import numpy as np
import matplotlib.pyplot as plt

def plot_residuals(y, yhat):
    """
    Create a residual plot.

    Args:
    - y: Actual target values.
    - yhat: Predicted target values.

    Returns:
    None
    """
    residuals = y - yhat
    plt.figure(figsize=(10, 6))
    plt.scatter(y, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Actual")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()

def regression_errors(y, yhat):
    """
    Calculate regression error metrics.

    Args:
    - y: Actual target values.
    - yhat: Predicted target values.

    Returns:
    - sse: Sum of Squared Errors
    - ess: Explained Sum of Squares
    - tss: Total Sum of Squares
    - mse: Mean Squared Error
    - rmse: Root Mean Squared Error
    """
    sse = ((y - yhat) ** 2).sum()
    ess = ((yhat - y.mean()) ** 2).sum()
    tss = ((y - y.mean()) ** 2).sum()
    mse = sse / len(y)
    rmse = np.sqrt(mse)
    return sse, ess, tss, mse, rmse

def baseline_mean_errors(y):
    """
    Calculate regression error metrics for the baseline model.

    Args:
    - y: Actual target values.

    Returns:
    - sse_baseline: SSE for the baseline model
    - mse_baseline: MSE for the baseline model
    - rmse_baseline: RMSE for the baseline model
    """
    y_baseline = np.mean(y)
    sse_baseline = ((y - y_baseline) ** 2).sum()
    mse_baseline = sse_baseline / len(y)
    rmse_baseline = np.sqrt(mse_baseline)
    return sse_baseline, mse_baseline, rmse_baseline

def better_than_baseline(y, yhat):
    """
    Determine if the model performs better than the baseline.

    Args:
    - y: Actual target values.
    - yhat: Predicted target values.

    Returns:
    - True if the model performs better than the baseline, otherwise False.
    """
    sse, _, _, _, _ = regression_errors(y, yhat)
    sse_baseline, _, _ = baseline_mean_errors(y)
    return sse < sse_baseline  # Model is better if its SSE is lower than baseline SSE

# -


