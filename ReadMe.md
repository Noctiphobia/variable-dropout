# Variable dropout

Package can be used to establish the features importance for any classifiaction or regression model. 

# Method description
The functionality is based on the comparison of the loss function value for two models. The first model is the full one. In the second values for the one variable are shuffled.  

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
```sh
rng = random.RandomState(0)
X = pd.DataFrame({'linear': rng.randint(-5, 5, 2000),
                  'square': rng.randint(-5, 5, 2000),
                  'noise': rng.randint(-5, 5, 2000)})
                  
y = [(row.square**2 - 2*row.linear + 1 + 0.1 * rng.randn()) for row in X.itertuples()]
y = [val > np.mean(y) for val in y]
```
Prepare model:
```sh
model = LogisticRegression(random_state=rng)
model.fit(X, y)
```
Calculate importance:
```sh
importance = variable_dropout_loss(model, X, y, loss_function=hinge_loss, random_state=rng)
```
Results:
```sh
importance
```

_baseline_      1.01624
linear          0.98859
square          0.82997
noise           0.81722
_full_model_    0.81736



