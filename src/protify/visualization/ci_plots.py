import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

from visualization.pauc_plot import plot_roc_with_ci


def regression_ci_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | os.PathLike[str],
    title: str | None = None,
) -> None:
    """
    Calculate the spearman rho and p-value of the regression model.
    Plot the line of best fit with 95% confidence intervals for spearman rho.
    Display the R-squared value, spearman rho, pearson rho, and p-values.
    """
    # Inputs may have any rank; n is the shared flattened element count.
    # y_true: (...); y_pred: (...)
    y_true, y_pred = y_true.flatten(), y_pred.flatten()  # (n,), (n,)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)  # (n,); n_f = finite paired observations
    y_true, y_pred = y_true[mask], y_pred[mask]  # (n_f,), (n_f,)
    r2 = r2_score(y_true, y_pred)
    r_s, p_s = spearmanr(y_true, y_pred)
    r_p, p_p = pearsonr(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, ax=ax)
    sns.regplot(
        x=y_true, y=y_pred,
        ci=95, ax=ax, scatter=False,
        line_kws={'color': 'red'}
    )

    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Regression Plot with 95% Confidence Interval')

    stats_text = (
        f"$R^2$ = {r2:.2f}\n"
        f"Spearman $\\rho$ = {r_s:.2f}  (p = {p_s:.2e})\n"
        f"Pearson $\\rho$ = {r_p:.2f}  (p = {p_p:.2e})"
    )
    ax.text(
        0.05, 0.95, stats_text,
        transform=ax.transAxes,
        fontsize=12, verticalalignment='top'
    )

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def classification_ci_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | os.PathLike[str],
    title: str | None = None,
) -> None:
    """Plot a classification ROC curve with confidence intervals."""
    # y_true: (n,), (b, l), or (n, c); y_pred: (n,), (b, l, c), or (n, c)
    if len(y_pred.shape) == 3 and len(y_true.shape) == 2:
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])  # (b * l, c)
        y_true = y_true.reshape(-1)  # (b * l,)

    # Flattening intentionally produces one aggregate multilabel plot.
    if len(y_pred.shape) == 2 and len(y_true.shape) == 2:
        y_pred = y_pred.flatten()  # (n * c,)
        y_true = y_true.flatten()  # (n * c,)

    # Cap the leading sample dimension because pAUC can be slow.
    if y_true.shape[0] > 10000:
        y_pred = y_pred[:10000]  # (n_cap, ...), n_cap = 10_000
        y_true = y_true[:10000]  # (n_cap, ...), n_cap = 10_000

    print(y_true.shape, y_pred.shape)

    try:
        plot_roc_with_ci(y_true, y_pred, save_path, fig_title=title)
    except Exception as error:
        print(f"Error plotting pAUC curve, likely the wrong version: {error}")


if __name__ == "__main__":
    # py -m visualization.ci_plots
    os.makedirs("plots/test_plots", exist_ok=True)
    y_true = np.random.rand(100)  # (100,)
    y_pred = np.random.rand(100)  # (100,)
    regression_ci_plot(y_true, y_pred, "plots/test_plots/regression.png", title="Regression Plot")

    y_true = np.random.randint(0, 2, (50, 514))  # (50, 514)
    y_pred = np.random.rand(50, 514, 4)  # (50, 514, 4)
    classification_ci_plot(y_true, y_pred, "plots/test_plots/classification.png", title="Classification Plot")
