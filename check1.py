
# coding: utf-8

# In[19]:


from functools import partial
import pandas as pd
from typing import Callable, Iterable, Any, Union, Optional, Tuple, List
from operator import itemgetter
from enum import Enum
from numpy import random
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt


class DropoutType(Enum):
    """
    Method of variable dropout loss representation. One of the following:
        RAW - raw value of variable dropout loss
        RATIO - ratio of loss of variable dropout to loss for unperturbed model
        DIFFERENCE - difference between variable dropout loss and unperturbed model loss
    """
    RAW = (lambda loss, loss_0: loss,)
    RATIO = (lambda loss, loss_0: loss / loss_0,)
    DIFFERENCE = (lambda loss, loss_0: loss - loss_0,)

    def __call__(self, *args, **kwargs):
        return self.value[0](*args, **kwargs)


def variable_dropout_loss(estimator: Any, X: pd.DataFrame, y: Iterable[Any],
                          loss_function: Callable[[Iterable[Any], Iterable[Any]], float] = mean_squared_error,
                          dropout_type: DropoutType = DropoutType.RAW, n_sample: int = 1000, n_iters: int = 100,
                          random_state: Optional[Union[int, random.RandomState]] = None) -> pd.Series:
    """
    Determines importance of variables in the model.
    Model trained on all variables is used to predict result variable for data
    with one variable randomly shuffled. The worse the result with a particular
    variable shuffled is, the more important the variable is.
    :param estimator: any fitted classification or regression model
                      with predict method.
    :param X: samples.
    :param y: result variable for samples.
    :param loss_function: a function taking vectors of real and predicted
                          results. The better the prediction,
                          the smaller the returned value.
    :param dropout_type: method of loss representation. One of values
                         specified in DropoutType enumeration.
    :param n_sample: number of samples to predict for. Given number of samples.
                     is randomly chosen from X with replacement.
    :param n_iters: number of iterations. Final result is mean of the results
                    of iterations.
    :param random_state: ensures deterministic results if run twice with the
                         same value.
    :return: series of variable dropout loss sorted descending.
    """
    y = list(y)
    _check_args(estimator, X, y, n_iters)
    rng = check_random_state(random_state)
    result = _single_variable_dropout(estimator, X, y, loss_function, dropout_type, n_sample, rng)
    for _ in range(n_iters - 1):
        result += _single_variable_dropout(estimator, X, y, loss_function, dropout_type, n_sample, rng)
    return result / n_iters


def _single_variable_dropout(estimator: Any, X: pd.DataFrame, y: List[Any],
                          loss_function: Callable[[Iterable[Any], Iterable[Any]], float],
                          dropout_type: DropoutType, n_sample: int, rng: random.RandomState) -> pd.Series:
    sampled_X, sampled_y = _sample_data(X, y, n_sample, rng)
    loss_0 = loss_function(sampled_y, estimator.predict(sampled_X))
    loss_full = loss_function(_shuffle(sampled_y, rng), estimator.predict(sampled_X))
    dropout_function = partial(dropout_type, loss_0=loss_0)
    results = {}
    for column in sampled_X.columns:
        perturbed_X = sampled_X.copy()
        perturbed_X[column] = _shuffle(list(perturbed_X[column]), rng)
        results[column] = dropout_function(loss_function(sampled_y, estimator.predict(perturbed_X)))
    keys = sorted(results, key=results.get, reverse=True)
    values = [dropout_function(loss_full), *[results[key] for key in keys], dropout_function(loss_0)]
    return pd.Series(data=values, index=['_baseline_', *keys, '_full_model_'])


def _check_args(estimator: Any, X: pd.DataFrame, y: List[Any], n_iters: int) -> None:
    if not hasattr(estimator, 'predict'):
        raise ValueError('Estimator does not have a predict method.')
    if len(X.columns) == 0:
        raise ValueError('X does not have any columns.')
    if len(X) != len(y):
        raise ValueError('Length of X does not match length of y.')
    if n_iters <= 0:
        raise ValueError('n_iters must be positive.')


def _sample_data(X: pd.DataFrame, y: List[Any], n_sample: int, rng: random.RandomState) ->         Tuple[pd.DataFrame, List[Any]]:
    if n_sample <= 0:
        return X, y
    else:
        indices = rng.choice(range(len(X)), n_sample, replace=True)
        return X.iloc[indices, :], itemgetter(*indices)(y)


def _shuffle(y: List[Any], rng: random.RandomState) -> List[Any]:
    return rng.choice(y, len(y), replace=False)



def plot_variable_dropout_loss(*args : pd.Series, maxvars = 10, figsize = (8, 10)):
       
    #print(len(args))
    
    indexes = ((list(args)[0])).index

    index_dictionary = {}

    for ind in indexes:
        tmp = [x[ind] for x in args]
        index_dictionary[ind] = (np.mean(tmp), np.max(tmp))
        
    sorted_indexes = sorted(index_dictionary.items(), key = lambda x: x[1], reverse = True) 
    
    if(maxvars != None):
        sorted_indexes = sorted_indexes[0:maxvars]
        
    maxx = np.max([x[1][1] for x in sorted_indexes])
    
    maxx *= 1.05
    
    selected_indexes = [x[0] for x in sorted_indexes]
    
    plots_number = len(selected_indexes)
    
    counter = 0  
    
    fig = plt.figure(figsize=figsize)
    fig.suptitle("drop-out loss", y=1.02, fontsize=12)
    
    for arg in args:
            
        counter +=1       
            
        tmp = plt.subplot(plots_number, 1, counter)

        tmp.set_xlim([0, maxx])
        
        labels = selected_indexes
        
        y_pos = np.arange(len(labels))
        
        values = arg[labels]  
        
        xerr = []
        
        for v in values:
            xerr.append((v, 0))
            
        xerr = np.array(xerr).T  
        
        plt.barh(y_pos, values, [0]*len(labels), yerr=0.3, xerr = xerr, ecolor='grey')
        
        tmp.set_yticks(y_pos)
        tmp.set_yticklabels(labels)
        
        title = 'model ' + str(counter)        
        
        tmp.set_title(title)

        plt.gca().invert_yaxis()   

            
    plt.tight_layout()

    plt.show()    


# In[140]:


import numpy as np

rng = random.RandomState(0)
X = pd.DataFrame({'linear': rng.randint(-5, 5, 2000),
                  'square': rng.randint(-5, 5, 2000),
                  'noise': rng.randint(-5, 5, 2000)})
                  
y = [(row.square**2 - 2*row.linear + 1 + 0.1 * rng.randn()) for row in X.itertuples()]
y = [val > np.mean(y) for val in y]


from sklearn.linear_model import LogisticRegression
import sklearn as sk


model = LogisticRegression(random_state=rng)
model.fit(X, y)


importance = variable_dropout_loss(model, X, y, loss_function=sk.metrics.hinge_loss, random_state=rng)

plot_variable_dropout_loss(importance, maxvars = 10)


# In[188]:


from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

#import xgboost as xgb

dataset = datasets.load_breast_cancer()

X = pd.DataFrame(dataset.data)
Y=dataset['target']




model_rf = RandomForestClassifier()
model_rf.fit(X = X, y=Y)

importance_rf = variable_dropout_loss(model_rf, X, Y, loss_function=sk.metrics.hinge_loss, random_state=rng)

#porównanie istotności z lasu losowego i funkcji

s = model_rf.feature_importances_

imp = sorted(range(len(s)), key=lambda k: s[k], reverse = True)

vals = s[imp]

imps = pd.DataFrame(vals, imp)

print("Random Forest importance")
print(imps)

print("dropout importance")

print(importance_rf)

plot_variable_dropout_loss(importance_rf, maxvars = 10, figsize = (15, 15))

#another model 

model_lr = LogisticRegression()

model_lr.fit(X,y=Y)

importance_lr = variable_dropout_loss(model_lr, X, Y, loss_function=sk.metrics.hinge_loss, random_state=rng)

#model comparison

plot_variable_dropout_loss(importance_rf, importance_lr, maxvars = 10, figsize = (20, 20))

