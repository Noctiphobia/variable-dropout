import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional


def plot_variable_dropout(*args: pd.DataFrame, max_vars: Optional[int] = 10, include_baseline_and_full: bool = True) -> None:
    """
    Plots the results of variable_dropout.

    :param args: any number of variable_dropout results.
    :param max_vars: maximum number of variables to plot per classifier, or None to plot all of them.
    :param include_baseline_and_full: whether to include _baseline_ and _full_model_ in the plot.
    """
    fig = plt.figure()
    fig.suptitle('Dropout loss', y=1.0)
    max_x = max(max(float(x) for x in importance['dropout_loss']) for importance in args) * 1.15
    for counter, arg in enumerate(args):
        model_name = arg['label'][0]
        arg = pd.Series([float(x) for x in arg['dropout_loss']], list(arg['variable']))
        subplot = plt.subplot(len(args), 1, counter + 1)
        subplot.set_xlim([0, max_x])
        subplot.set_title(model_name)
        values = _get_data_to_plot(arg, max_vars, include_baseline_and_full)[::-1]
        full_model_loss = arg['_full_model_']
        values_to_plot = values - full_model_loss
        ax = values_to_plot.plot.barh(color='grey', edgecolor='black', left=full_model_loss)
        for p in ax.patches:
            ax.annotate(str(round(p.get_x() + p.get_width(), 2)),
                        ((p.get_x() + p.get_width()) * 1.01, p.get_y() * 1.01))
    plt.tight_layout()
    plt.show()


def _get_data_to_plot(importance: pd.Series, max_vars: Optional[int], include_baseline_and_full: bool) -> pd.Series:
    if max_vars is None or max_vars > len(importance) - 2:
        max_vars = len(importance) - 2
    result = importance[1:(max_vars + 1)]
    if include_baseline_and_full:
        result = importance[:1].append([result, importance[-1:]])
    return result
