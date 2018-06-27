# Variable dropout

Package provides an implementation of variable dropout method, which can be used to establish the importance of features for any classification or regression model. For full description, visit [the documentation](https://variable-dropout.readthedocs.io/en/latest/).

# Method description
The functionality is based on the comparison of the loss function value for two models. The first model is the full one. In the second values for the one variable are shuffled.

# Installation
```sh
git clone https://github.com/Noctiphobia/variable-dropout/
cd variable-dropout
python setup.py install
```
If ```python``` resolves to Python 2, replace it with ```python3```.

# Requirements
  - Python 3.6
  - scikit-learn 0.19.1
  - numpy 1.11.3
  - pandas 0.22.0
  - matplotlib 2.2.0

# Arguments
- param estimator: any fitted classification or regression model
                      with predict method.
- param X: samples.
- param y: result variable for samples.
- param loss_function: a function taking vectors of real and predicted results. The better the prediction, the smaller the returned value.
- param dropout_type: method of loss representation. One of values specified in DropoutType enumeration.
- param n_sample: number of samples to predict for. Given number of samples is randomly chosen from X with replacement.
- param n_iters: number of iterations. Final result is mean of the results of iterations.
- param random_state: ensures deterministic results if run twice with the same value.
- return: series of variable dropout loss sorted descending.

# Basic usage
Load dataset and output:
```python
rng = random.RandomState(0)
X = pd.DataFrame({'linear': rng.randint(-5, 5, 2000),
                  'square': rng.randint(-5, 5, 2000),
                  'noise': rng.randint(-5, 5, 2000)})

y = [(row.square**2 - 2*row.linear + 1 + 0.1 * rng.randn()) for row in X.itertuples()]
y = [val > np.mean(y) for val in y]
```
Prepare model:
```python
model = LogisticRegression(random_state=rng)
model.fit(X, y)
```
Calculate importance:
```python
importance = variable_dropout(model, X, y, loss_function=hinge_loss, random_state=rng)
```
Results:
```python
importance

- _baseline_      1.01624
- linear          0.98859
- square          0.82997
- noise           0.81722
- _full_model_    0.81736
```

Real data example
-----------

### Load dataset

Import necessary packages.

```python
   from sklearn import datasets
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.linear_model import LogisticRegression
   import sklearn as sk
   import numpy as np
   import pandas as pd
   from xgboost import XGBClassifier
   from variable_dropout.variable_dropout import variable_dropout
   from variable_dropout.plot_variable_dropout import plot_variable_dropout
```

Import data, extract input dataset and a target vector.

```python

   dataset = datasets.load_breast_cancer()
   data = pd.DataFrame(dataset.data)
   target = dataset.target
   data.columns = dataset.feature_names
```


### Prepare models


Create a classification or a regression model.

```python

   model_rf = RandomForestClassifier(random_state=0)
   model_lr = LogisticRegression(random_state=0)
   model_xgb = XGBClassifier(random_state=0)
```

### Train models

Train models on data.

```python
   model_rf.fit(X = data, y = target)
   model_lr.fit(X = data, y = target)
   model_xgb.fit(X = data, y = target)
```

### Get features importance

Compute features importance for models.

```python
   importance_rf = variable_dropout(model_rf, data, target, loss_function=sk.metrics.hinge_loss, random_state=0)
   importance_lr = variable_dropout(model_lr, data, target, loss_function=sk.metrics.hinge_loss, random_state=0)
   importance_xgb = variable_dropout(model_xgb, data, target, loss_function=sk.metrics.hinge_loss, random_state=0)
```


### Text form of importance

Display computed importance for a model.

```python
   importance_rf
::


	                   variable  dropout_loss                   label
	0                _baseline_       0.84016  RandomForestClassifier
	1                area error       0.39089  RandomForestClassifier
	2              worst radius       0.38959  RandomForestClassifier
	3       mean concave points       0.38006  RandomForestClassifier
	4              mean texture       0.38068  RandomForestClassifier
	5             worst texture       0.37935  RandomForestClassifier
	6      worst concave points       0.38272  RandomForestClassifier
	7           worst concavity       0.37685  RandomForestClassifier
	8           worst perimeter       0.37748  RandomForestClassifier
	9            mean perimeter       0.37744  RandomForestClassifier
	10           mean concavity       0.37591  RandomForestClassifier
	11               worst area       0.37906  RandomForestClassifier
	12         worst smoothness       0.37760  RandomForestClassifier
	13          mean smoothness       0.37539  RandomForestClassifier
	14              mean radius       0.37618  RandomForestClassifier
	15            mean symmetry       0.37454  RandomForestClassifier
	16          concavity error       0.37489  RandomForestClassifier
	17   mean fractal dimension       0.37485  RandomForestClassifier
	18             radius error       0.37592  RandomForestClassifier
	19          perimeter error       0.37539  RandomForestClassifier
	20     concave points error       0.37482  RandomForestClassifier
	21         mean compactness       0.37410  RandomForestClassifier
	22            texture error       0.37410  RandomForestClassifier
	23         smoothness error       0.37424  RandomForestClassifier
	24        compactness error       0.37410  RandomForestClassifier
	25           symmetry error       0.37426  RandomForestClassifier
	26  fractal dimension error       0.37413  RandomForestClassifier
	27        worst compactness       0.37467  RandomForestClassifier
	28           worst symmetry       0.37426  RandomForestClassifier
	29  worst fractal dimension       0.37434  RandomForestClassifier
	30                mean area       0.37381  RandomForestClassifier
	31             _full_model_       0.37410  RandomForestClassifier

```


### Visual form of importance

Visualize importance for one model.

```python
   plot_variable_dropout(importance_rf)
```

![one-model](https://github.com/Noctiphobia/variable-dropout/blob/master/Sphinx/pic1.png)

Visualize importance for multiple models.

```python

   plot_variable_dropout(importance_rf, importance_lr,importance_xgb)
```

![multi-models](https://github.com/Noctiphobia/variable-dropout/blob/master/Sphinx/pic2.png)



