Usage
=====
Package can be used to establish the features importance for any classifiaction or regression model.

Method description
------------------
The functionality is based on the comparison of the loss function value for two models. The first model is the full one. In the second values for the one variable are shuffled.

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

.. code:: python
	
   from sklearn import datasets
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.linear_model import LogisticRegression
   import sklearn as sk
   import numpy as np
	

.. code:: python

   rng = random.RandomState(0)
   dataset = datasets.load_breast_cancer()
	
.. code:: python
	
   X = pd.DataFrame(dataset.data)
   Y=dataset['target']


Prepare model
~~~~~~~~~~~~~

.. code:: python

   model_rf = RandomForestClassifier()
   model_lr = LogisticRegression()
   model_xgb = XGBClassifier()

Train model
~~~~~~~~~~~

.. code:: python

   model_rf.fit(X = X, y=Y)
   model_lr.fit(X,y=Y)
   model_xgb.fit(X = X, y=Y)


Get features importance
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   importance_rf = variable_dropout(model_rf, X, Y, loss_function=sk.metrics.hinge_loss, random_state=rng)
   importance_lr = variable_dropout(model_lr, X, Y, loss_function=sk.metrics.hinge_loss, random_state=rng)
   importance_xgb = variable_dropout(model_xgb, X, Y, loss_function=sk.metrics.hinge_loss, random_state=rng)
   


Text form of importance
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   importance_rf

::

               
     _baseline_      0.84351
	23              0.39114
	22              0.39178
	7               0.38291
	13              0.38380
	19              0.38221
	21              0.38623
	27              0.38392
	12              0.38529
	1               0.38391
	2               0.38180
	3               0.38081
	6               0.38238
	15              0.38013
	10              0.38089
	11              0.38109
	14              0.38214
	16              0.38150
	18              0.38048
	24              0.37999
	25              0.38108
	26              0.38002
	0               0.38183
	4               0.38032
	5               0.38000
	8               0.37943
	9               0.37945
	17              0.37945
	20              0.38238
	28              0.38014
	29              0.37957
	_full_model_    0.37945             
 


Visual form of importance
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   plot_variable_dropout(importance_rf, maxvars = 10, figsize = (20, 20))


.. figure:: pic1.png
   :alt: png

   
   It is also possible to place more importance sets on the one chart.
 
   
.. code:: python

   plot_variable_dropout(importance_rf, importance_lr,importance_xgb, maxvars = 10, figsize = (20, 20))
   
.. figure:: pic2.png
   :alt: png

   


