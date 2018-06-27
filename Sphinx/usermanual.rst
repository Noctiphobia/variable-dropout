Usage
=====
Package can be used to establish the features importance for any classifiaction or regression model. You can present the result in text and graphical form. Additonally, you can present a few plots on the one graphics.

Method description
------------------
The functionality is based on the comparison of the loss function value for two models. The first model is the full one. In the second, values for the one variable are shuffled. In order to minimize the influence of randomness, the number of algorithm iteration was added. The final result is obtained as the mean of the results from the all iterations.

Installation
------------

.. code:: sh

   git clone https://github.com/Noctiphobia/variable-dropout/
   cd variable-dropout
   python setup.py install

If python resolves to Python 2, replace it with python3.

Requirements
------------

- python 3.6
- scikit-learn 0.19.1 
- numpy 1.11.3
- pandas 0.22.0
- matplotlib 2.2.0


Basic usage
-----------

Load dataset
~~~~~~~~~~~~

Import necessary packages.

.. code:: python
	
   from sklearn import datasets
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.linear_model import LogisticRegression
   import sklearn as sk
   import numpy as np
   import pandas as pd
   from xgboost import XGBClassifier
   from variable_dropout.variable_dropout import variable_dropout
   from variable_dropout.plot_variable_dropout import plot_variable_dropout
	

Import data, extract input dataset and a target vector.
	

.. code:: python

   dataset = datasets.load_breast_cancer()
   data = pd.DataFrame(dataset.data)
   target = dataset.target
   data.columns = dataset.feature_names
   


Prepare models
~~~~~~~~~~~~~~~

Create a classification or a regression model.

.. code:: python

   model_rf = RandomForestClassifier(random_state=0)
   model_lr = LogisticRegression(random_state=0)
   model_xgb = XGBClassifier(random_state=0)

Train models
~~~~~~~~~~~~~

Train models on data.

.. code:: python

   model_rf.fit(X = data, y = target)
   model_lr.fit(X = data, y = target)
   model_xgb.fit(X = data, y = target)


Get features importance
~~~~~~~~~~~~~~~~~~~~~~~

Compute features importance for models.

.. code:: python

   importance_rf = variable_dropout(model_rf, data, target, loss_function=sk.metrics.hinge_loss, random_state=0)
   importance_lr = variable_dropout(model_lr, data, target, loss_function=sk.metrics.hinge_loss, random_state=0)
   importance_xgb = variable_dropout(model_xgb, data, target, loss_function=sk.metrics.hinge_loss, random_state=0)
   


Text form of importance
~~~~~~~~~~~~~~~~~~~~~~~~

Display computed importance for a model.

.. code:: python

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

               
  
    

Visual form of importance
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Visualize importance for one model.

.. code:: python

   plot_variable_dropout(importance_rf)


.. figure:: pic1.png
   :alt: png

   
   
Visualize importance for multiple models.

   
.. code:: python

   plot_variable_dropout(importance_rf, importance_lr,importance_xgb)
   
.. figure:: pic2.png
   :alt: png

   


