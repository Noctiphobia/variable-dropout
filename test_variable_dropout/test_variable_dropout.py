from unittest import TestCase
from numpy import random
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import hinge_loss

from variable_dropout.variable_dropout import variable_dropout


class TestVariableDropout(TestCase):

    def test_random_noise_least_important_for_regression(self):
        rng = random.RandomState(0)
        X = pd.DataFrame({'linear': rng.randint(-5, 5, 2000),
                          'square': rng.randint(-5, 5, 2000),
                          'noise': rng.randint(-5, 5, 2000)})
        y = [row.square**2 - 2*row.linear + 1 + 0.1 * rng.randn() for row in X.itertuples()]
        model = LinearRegression(n_jobs=-1)
        model.fit(X, y)
        importance = variable_dropout(model, X, y, random_state=rng)
        self.assertGreater(importance['linear'], importance['noise'])
        self.assertGreater(importance['square'], importance['noise'])

    def test_random_noise_least_important_for_classification(self):
        rng = random.RandomState(0)
        X = pd.DataFrame({'linear': rng.randint(-5, 5, 2000),
                          'square': rng.randint(-5, 5, 2000),
                          'noise': rng.randint(-5, 5, 2000)})
        y = [(row.square**2 - 2*row.linear + 1 + 0.1 * rng.randn()) for row in X.itertuples()]
        y = [val > np.mean(y) for val in y]
        model = LogisticRegression(random_state=rng)
        model.fit(X, y)
        importance = variable_dropout(model, X, y, loss_function=hinge_loss, random_state=rng)
        self.assertGreater(importance['linear'], importance['noise'])
        self.assertGreater(importance['square'], importance['noise'])

    def test_same_results_with_same_random_state(self):
        rng = random.RandomState(0)
        X = pd.DataFrame({'linear': rng.randint(-5, 5, 2000),
                          'square': rng.randint(-5, 5, 2000),
                          'noise': rng.randint(-5, 5, 2000)})
        y = [(row.square ** 2 - 2 * row.linear + 1 + 0.1 * rng.randn()) for row in X.itertuples()]
        y = [val > np.mean(y) for val in y]
        model = LogisticRegression(random_state=rng)
        model.fit(X, y)
        importance_first = variable_dropout(model, X, y, n_iters=1, loss_function=hinge_loss, random_state=0)
        importance_second = variable_dropout(model, X, y, n_iters=1, loss_function=hinge_loss, random_state=0)
        self.assertEqual(list(importance_first), list(importance_second))

    def test_bad_estimator(self):
        with self.assertRaises(ValueError):
            variable_dropout([], pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}), [0, 1, 2])

    def test_empty_data(self):
        with self.assertRaises(ValueError):
            variable_dropout(LinearRegression(), pd.DataFrame({'a': [], 'b': []}), [])

    def test_x_longer(self):
        with self.assertRaises(ValueError):
            variable_dropout(LinearRegression(), pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}), [0, 1])

    def test_y_longer(self):
        with self.assertRaises(ValueError):
            variable_dropout(LinearRegression(), pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}), [0, 1, 2, 3])

    def test_negative_iters(self):
        with self.assertRaises(ValueError):
            variable_dropout(LinearRegression(), pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}), [0, 1, 2],
                             n_iters=-1)
