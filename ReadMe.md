# Variable dropout

Package provides an implementation of variable dropout method, which can be used to establish the importance of features for any classification or regression model. For full description, visit [the documentation](https://variable-dropout.readthedocs.io/en/latest/).

# Method description
The functionality is based on the comparison of the loss function value for two models. The first model is the full one. In the second values for the one variable are pythonuffled.

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
   from xgboost import XGBClassifier
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

   model_rf = RandomForestClassifier()
   model_lr = LogisticRegression()
   model_xgb = XGBClassifier()
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
   importance_rf = variable_dropout(model_rf, data, target, loss_function=sk.metrics.hinge_loss, random_state=rng)
   importance_lr = variable_dropout(model_lr, data, target, loss_function=sk.metrics.hinge_loss, random_state=rng)
   importance_xgb = variable_dropout(model_xgb, data, target, loss_function=sk.metrics.hinge_loss, random_state=rng)
```


### Text form of importance

Display computed importance for a model.

```python
   importance_rf
::


	0                _baseline_       0.84005  RandomForestClassifier
	1                worst area       0.44505  RandomForestClassifier
	2           worst perimeter       0.40254  RandomForestClassifier
	3          worst smoothness       0.37953  RandomForestClassifier
	4            mean concavity       0.37759  RandomForestClassifier
	5       mean concave points       0.38145  RandomForestClassifier
	6             worst texture       0.37810  RandomForestClassifier
	7          mean compactness       0.37568  RandomForestClassifier
	8              mean texture       0.37818  RandomForestClassifier
	9              radius error       0.37554  RandomForestClassifier
	10     worst concave points       0.37553  RandomForestClassifier
	11              mean radius       0.37478  RandomForestClassifier
	12           mean perimeter       0.37550  RandomForestClassifier
	13         smoothness error       0.37595  RandomForestClassifier
	14          worst concavity       0.37503  RandomForestClassifier
	15                mean area       0.37523  RandomForestClassifier
	16          mean smoothness       0.37425  RandomForestClassifier
	17            mean symmetry       0.37425  RandomForestClassifier
	18   mean fractal dimension       0.37425  RandomForestClassifier
	19            texture error       0.37479  RandomForestClassifier
	20          perimeter error       0.37425  RandomForestClassifier
	21               area error       0.37512  RandomForestClassifier
	22        compactness error       0.37574  RandomForestClassifier
	23          concavity error       0.37425  RandomForestClassifier
	24     concave points error       0.37425  RandomForestClassifier
	25           symmetry error       0.37425  RandomForestClassifier
	26  fractal dimension error       0.37510  RandomForestClassifier
	27        worst compactness       0.37439  RandomForestClassifier
	28           worst symmetry       0.37464  RandomForestClassifier
	29  worst fractal dimension       0.37431  RandomForestClassifier
	30             worst radius       0.37306  RandomForestClassifier
	31             _full_model_       0.37425  RandomForestClassifier

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



