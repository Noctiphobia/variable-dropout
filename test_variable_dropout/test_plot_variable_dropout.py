from unittest import TestCase
import pandas as pd

from variable_dropout.plot_variable_dropout import _get_data_to_plot


class TestPlotVariableDropout(TestCase):
    importances = pd.Series([150, 120, 110, 60], ['_baseline_', 'a', 'b', '_full_model_'])

    def test_get_data_to_plot(self):
        result = _get_data_to_plot(self.importances, 1, True)
        self.assertEqual(3, len(result))
        self.assertEqual(120, result[1])

    def test_get_data_to_plot_all_variables(self):
        result = _get_data_to_plot(self.importances, None, False)
        self.assertEqual(2, len(result))

    def test_get_data_to_plot_only_base_full(self):
        result = _get_data_to_plot(self.importances, 0, True)
        self.assertEqual(2, len(result))

    def test_get_data_to_plot_empty(self):
        result = _get_data_to_plot(self.importances, 0, False)
        self.assertEqual(0, len(result))