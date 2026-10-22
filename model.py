"""
NumPy Multiple Linear Regression GD

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - shuffle_xy
import numpy as np

def shuffle_xy(X, y, seed=42):
    """Randomly permute feature rows and targets together.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Feature matrix.
    y : np.ndarray, shape (n,)
        Target vector.
    seed : int, optional
        RNG seed for reproducibility (default 42).

    Returns
    -------
    X_shuffled : np.ndarray, shape (n, d)
    y_shuffled : np.ndarray, shape (n,)
    """
    # TODO: Return (X, y) under one shared seeded row permutation

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(X.shape[0])

    X_new = X[permutation]
    y_new = y[permutation]

    return (X_new, y_new)

# Step 2 - split_train_val_test
def split_train_val_test(X, y, train_frac=0.6, val_frac=0.2):
    # TODO: Slice already-shuffled data into contiguous train/val/test partitions...
    
    n = X.shape[0]

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]

    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]

    return (X_train, y_train, X_val, y_val, X_test, y_test)

# Step 3 - compute_feature_stats
def compute_feature_stats(X):
    # TODO: Compute per-feature mean and std; replace std of 0 with 1

    p = X.shape[1]

    means = []
    stds = []

    for j in range(p):

        X_j = X[:, j]

        means.append(np.mean(X_j))
        std_ = np.std(X_j)

        if std_ == 0: std_ = 1.0
        stds.append(std_)

    return (means, stds)

# Step 4 - standardize_features
def standardize_features(X, mean, std):
    # TODO: Apply z-score normalization using precomputed training mean and std.

    for j in range(X.shape[1]):
        X[:,j] = (X[:,j] - mean[j])/std[j]

    return X

# Step 5 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to feature matrix X
    
    X = np.column_stack((np.ones(X.shape[0]), X))
    return X

# Step 6 - prepare_design_matrix
def prepare_design_matrix(X, mean, std):
    # TODO: Standardize features then add the bias column to form the design matrix.
    
    # standardizzo
    X = standardize_features(X, mean, std)

    # aggiungo bias column
    X = add_bias_column(X)

    return X

# Step 7 - predict_linear
def predict_linear(X, weights):
    """Compute linear predictions y_hat = X @ weights.

    Args:
        X: Design matrix of shape (n, d_in), often including a bias column.
        weights: Weight vector of shape (d_in,).

    Returns:
        Predicted targets of shape (n,).
    """
    # TODO: Return the predicted target vector from X and weights
    
    return X @ weights

# Step 8 - mse_loss
def mse_loss(y_true, y_pred):
    # TODO: Return the average of squared residuals as a scalar float.

    return np.mean((y_true - y_pred)**2)

# Step 9 - mse_gradient
def mse_gradient(X, y_true, y_pred):
    # TODO: Return the analytic MSE gradient w.r.t. weights: (2/n) X^T (y_pred - y_true)
    
    n = X.shape[0]
    return (2 / n) * X.T @ (y_pred - y_true)

# Step 10 - normal_equation
def normal_equation(X, y):
    # TODO: Solve for the closed-form least-squares weights via the normal equation.
    
    return np.linalg.solve(X.T @ X, X.T @ y)

# Step 11 - initialize_weights
def initialize_weights(n_features, seed=None):
    # TODO: Return (n_features,) weights sampled from N(0, 0.01)
    
    if seed is not None:
        np.random.seed(seed) 

    weights = np.random.normal(0, 0.01, size = n_features)
    
    return weights

# Step 12 - gd_step
def gd_step(X, y, weights, lr):
    """Run one full-batch gradient descent update on the weights.

    Args:
        X: Design matrix of shape (n, d_in).
        y: Target vector of shape (n,).
        weights: Current weight vector of shape (d_in,).
        lr: Learning rate (float).

    Returns:
        Updated weight vector of shape (d_in,).
    """
    # TODO: return the updated weight vector after one MSE gradient step
    
    # current predictions
    y_pred = predict_linear(X, weights)

    # MSE gradient
    grad = mse_gradient(X, y, y_pred)

    # update weights
    new_weights  = weights - lr * grad 

    return new_weights

# Step 13 - epoch_train_val_losses
def epoch_train_val_losses(X_train, y_train, X_val, y_val, weights):
    """Evaluate MSE on train and validation sets for the current weights.

    Args:
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        weights: Weight vector of shape (d_in,).

    Returns:
        (train_loss, val_loss) as plain floats.
    """
    # TODO: return the pair (train_loss, val_loss) as MSE floats
    
    y_pred_train = X_train @ weights
    y_pred_val = X_val @ weights

    train_loss = mse_loss(y_train, y_pred_train)
    val_loss = mse_loss(y_val, y_pred_val)

    return (train_loss, val_loss)

# Step 14 - update_early_stop_state
def update_early_stop_state(val_loss, best_val_loss, wait, weights, best_weights, patience):
    # TODO: Update best weights and patience counter; signal stop when val loss stalls...

    # "wait" count consecutive epochs without improvement
    # "patiente" is the wait limit 

    if val_loss < best_val_loss: 

        best_val_loss = val_loss
        best_weights = weights

        # resetto wait
        wait = 0

    else: wait+=1
    
    if wait == patience: 
        return best_val_loss, wait, best_weights, True 

    return best_val_loss, wait, best_weights, False

# Step 15 - init_training_state
def init_training_state(n_features, seed=None):
    # TODO: Build the initial training-state dictionary for the GD epoch loop.

    init_weights = initialize_weights(n_features, seed=seed)
    
    dict_ = {
        "weights": init_weights,
        "best_weights": init_weights.copy(),
        "best_val_loss": np.inf,
        "wait": 0,
        "train_losses": [],
        "val_losses": [],
        "stopped": False
    }

    return dict_

# Step 16 - run_one_epoch
def run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience):
    """Perform one GD step, log losses, and refresh early-stopping on state.

    Args:
        state: Dict with keys weights, best_weights, best_val_loss, wait,
            stopped, train_losses, val_losses.
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        lr: Learning rate (float).
        patience: Early-stopping patience (int).

    Returns:
        Updated state dict.
    """
    # TODO: Take one GD step, log train/val losses, refresh early-stopping fields...

    # make one step on the training set
    state["weights"] = gd_step(X=X_train, y=y_train, weights=state["weights"], lr=lr)

    train_loss, val_loss = epoch_train_val_losses(
        X_train, 
        y_train, 
        X_val, 
        y_val, 
        weights=state["weights"]
        )
    
    state["train_losses"].append(train_loss)
    state["val_losses"].append(val_loss)

    best_val_loss, wait, best_weights, stop = update_early_stop_state(
        val_loss=val_loss, 
        best_val_loss=state["best_val_loss"], 
        wait=state["wait"], 
        weights=state["weights"], 
        best_weights=state["best_weights"], 
        patience=patience
        )
    
    state["best_weights"] = best_weights
    state["best_val_loss"] = best_val_loss
    state["wait"] = wait
    state["stopped"] = stop

    return state

# Step 17 - train_batch_gd (not yet solved)
# TODO: implement

# Step 18 - mean_absolute_error (not yet solved)
# TODO: implement

# Step 19 - root_mean_squared_error (not yet solved)
# TODO: implement

# Step 20 - r_squared (not yet solved)
# TODO: implement

# Step 21 - evaluate_regression (not yet solved)
# TODO: implement

# Step 22 - learning_curve_data (not yet solved)
# TODO: implement

# Step 23 - weights_l2_distance (not yet solved)
# TODO: implement

# Step 24 - create_lr_model (not yet solved)
# TODO: implement

# Step 25 - fit_lr_model (not yet solved)
# TODO: implement

# Step 26 - predict_lr_model (not yet solved)
# TODO: implement

# Step 27 - score_lr_model (not yet solved)
# TODO: implement

# Step 28 - compare_with_normal_equation (not yet solved)
# TODO: implement

