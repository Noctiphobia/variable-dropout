import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional


def plot_variable_dropout(max_vars: Optional[int] = 10, include_baseline_and_full: bool = True, **kwargs: pd.Series) -> None:
    fig = plt.figure()
    fig.suptitle('Dropout loss', y=1.0)
    max_x = max(max(importance) for importance in kwargs.values()) * 1.1
    for counter, (model_name, arg) in enumerate(kwargs.items()):
        subplot = plt.subplot(len(kwargs), 1, counter + 1)
        subplot.set_xlim([0, max_x])
        subplot.set_title(model_name)
        values = _get_data_to_plot(arg, max_vars, include_baseline_and_full)[::-1]
        ax = values.plot.barh(color='grey')
        for p in ax.patches:
            ax.annotate(str(round(p.get_width(), 2)), (p.get_width() * 1.005, p.get_y() * 1.005))
    plt.tight_layout()
    plt.show()


def _get_data_to_plot(importance: pd.Series, max_vars: Optional[int], include_baseline_and_full: bool) -> pd.Series:
    if max_vars is None or max_vars > len(importance) - 2:
        max_vars = len(importance) - 2
    result = importance[1:(max_vars + 1)]
    if include_baseline_and_full:
        result = importance[:1].append([result, importance[-1:]])
    return result
