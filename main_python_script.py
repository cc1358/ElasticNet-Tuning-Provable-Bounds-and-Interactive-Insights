
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import ElasticNet
import streamlit as st
from joblib import Parallel, delayed

# --- Core ElasticNet Functions with AIC/BIC Adjustments ---
def elasticnet_loss(X, y, w, lambda1, lambda2, adjustment=None, p=None):
    """Compute ElasticNet validation loss with optional AIC/BIC adjustments."""
    m = X.shape[0]
    residual = y - X @ w
    mse = (1 / m) * np.sum(residual**2)
    reg_loss = mse + lambda1 * np.sum(np.abs(w)) + lambda2 * np.sum(w**2)
    
    if adjustment and p is not None:
        df = np.sum(w != 0)  # Degrees of freedom (non-zero coefficients)
        if adjustment == "AIC":
            return reg_loss + (2 * df) / m
        elif adjustment == "BIC":
            return reg_loss + (np.log(m) * df) / m
    return reg_loss

def fit_elasticnet(X, y, lambda1, lambda2):
    """Fit ElasticNet and return weights."""
    alpha = lambda1 + lambda2
    l1_ratio = lambda1 / alpha if alpha > 0 else 0.5
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False)
    model.fit(X, y)
    return model.coef_

def generate_problem(m=100, p=5, val_size=20):
    """Generate a regression problem with train and validation sets."""
    X = np.random.randn(m, p)
    w_true = np.random.randn(p)
    y = X @ w_true + 0.1 * np.random.randn(m)
    X_val = np.random.randn(val_size, p)
    y_val = X_val @ w_true + 0.1 * np.random.randn(val_size)
    return X, y, X_val, y_val

# --- Sample Complexity Tuning ---
def sample_complexity(p, epsilon, delta, H=1):
    """Compute sample complexity bound """
    return int((H**2 / epsilon**2) * (p**2 + np.log(1 / delta)))

def tune_elasticnet_batch(n, p, lambda_max=1.0, adjustment=None):
    """Batch tuning over n problem samples."""
    problems = [generate_problem(p=p) for _ in range(n)]
    lambda_range = np.linspace(0, lambda_max, 10)  # Reduced from 20 to 10
    losses = []
    
    def compute_loss(l1, l2):
        total_loss = 0
        for X, y, X_val, y_val in problems:
            w = fit_elasticnet(X, y, l1, l2)
            total_loss += elasticnet_loss(X_val, y_val, w, l1, l2, adjustment, p)
        return (total_loss / n, (l1, l2))
    
    # Parallelize the computation of losses
    losses = Parallel(n_jobs=-1)(delayed(compute_loss)(l1, l2) for l1 in lambda_range for l2 in lambda_range)
    
    avg_loss, best_lambda = min(losses, key=lambda x: x[0])
    _, opt_loss = min([tune_elasticnet_batch(100, p, lambda_max, adjustment)[0] for _ in range(3)], key=lambda x: x[0])
    return avg_loss, best_lambda, avg_loss - opt_loss

# --- Online Learning with EXP3 Bandit ---
def online_tune_elasticnet_exp3(T, p, lambda_max=1.0, gamma=0.1):
    """Online tuning with EXP3 bandit algorithm for Õ(√T) regret."""
    K = 5  # Reduced from 10 to 5
    lambda_grid = np.linspace(0, lambda_max, K)
    weights = np.ones((K, K)) / (K * K)  # Uniform initial weights over 2D grid
    regrets = []
    losses = []
    best_loss_history = []
    eta = np.sqrt(2 * np.log(K * K) / (T * K * K))  # Learning rate
    
    for t in range(T):
        # Sample λ from distribution
        probs = weights / weights.sum()
        i, j = np.unravel_index(np.random.choice(K * K, p=probs.flatten()), (K, K))
        lambda1, lambda2 = lambda_grid[i], lambda_grid[j]
        
        # Evaluate loss
        X, y, X_val, y_val = generate_problem(p=p)
        w = fit_elasticnet(X, y, lambda1, lambda2)
        loss_t = elasticnet_loss(X_val, y_val, w, lambda1, lambda2)
        losses.append(loss_t)
        
        # Compute best loss for regret
        best_loss = float('inf')
        for l1 in lambda_grid:
            for l2 in lambda_grid:
                w_opt = fit_elasticnet(X, y, l1, l2)
                opt_loss = elasticnet_loss(X_val, y_val, w_opt, l1, l2)
                best_loss = min(best_loss, opt_loss)
        regrets.append(loss_t - best_loss)
        best_loss_history.append(best_loss)
        
        # Update weights (EXP3 update rule)
        loss_scaled = loss_t / lambda_max  # Normalize to [0, 1]
        weights[i, j] *= np.exp(-eta * loss_scaled / probs[i, j])
    
    return np.cumsum(regrets), losses, best_loss_history

# --- Compute 1/2-Dispersion Using Function Norms ---
def compute_dispersion(losses, lambda_max=1.0):
    """Compute 1/2-dispersion using L2 norm of loss differences."""
    T = len(losses)
    norm_diffs = []
    for i in range(T):
        for j in range(i + 1, T):
            diff = losses[i] - losses[j]
            norm_diffs.append(np.abs(diff))  # L2 norm of scalar difference
    avg_norm_diff = np.mean(norm_diffs)
    # Check if average difference meets 1/2-dispersion threshold (heuristic)
    dispersion_threshold = 0.5 * lambda_max  # Assuming losses are scaled by lambda_max
    is_half_dispersed = avg_norm_diff >= dispersion_threshold
    return avg_norm_diff, is_half_dispersed

# --- Piecewise Decomposable Structure ---
def f_q(lambda1, lambda2, p=2):
    """Rational polynomial for F (degree ≤ 2p)."""
    q1 = lambda1**2 + 2 * lambda1 * lambda2 + lambda2**2
    q2 = lambda2**2 + 1
    return q1 / q2

def g_r(lambda1, lambda2, p=2):
    """Threshold polynomial for G (degree 1 in λ1, ≤ p in λ2)."""
    r = lambda1 + lambda2**p
    return 1 if r < 0 else 0

def compute_loss_surface(X, y, X_val, y_val, lambda_max=1.0, p=2):
    """Compute loss surface and G regions."""
    lambda_range = np.linspace(0, lambda_max, 10)  # Reduced from 20 to 10
    L = np.zeros((len(lambda_range), len(lambda_range)))
    G = np.zeros_like(L)
    
    for i, l1 in enumerate(lambda_range):
        for j, l2 in enumerate(lambda_range):
            w = fit_elasticnet(X, y, l1, l2)
            L[i, j] = elasticnet_loss(X_val, y_val, w, l1, l2)
            G[i, j] = g_r(l1, l2, p)
    
    return L, G, lambda_range

# --- Multi-Task Learning ---
def multi_task_elasticnet(problems, lambda1, lambda2, adjustment=None):
    """Fit ElasticNet across multiple tasks."""
    weights = []
    losses = []
    for X, y, X_val, y_val in problems[:3]:  # Reduced from 5 to 3
        w = fit_elasticnet(X, y, lambda1, lambda2)
        weights.append(w)
        losses.append(elasticnet_loss(X_val, y_val, w, lambda1, lambda2, adjustment, problems[0][0].shape[1]))
    return weights, np.mean(losses)

# --- Visualization and Main Execution ---
def plot_results(n, p, T, epsilon, delta, lambda_max=1.0, adjustment=None):
    """Generate all plots for portfolio showcase."""
    plt.figure(figsize=(15, 10))

    # Batch Tuning 
    n_bound = sample_complexity(p, epsilon, delta)
    n_range = [10, 50, 100, n_bound]
    loss_gaps = []
    for n in n_range:
        avg_loss, best_lambda, gap = tune_elasticnet_batch(n, p, lambda_max, adjustment)
        loss_gaps.append(gap)
    
    plt.subplot(2, 2, 1)
    plt.plot(n_range, loss_gaps, marker='o', label='Empirical Loss Gap')
    plt.axhline(epsilon, color='r', linestyle='--', label=f'ε = {epsilon}')
    plt.axvline(n_bound, color='g', linestyle='--', label=f'n = {n_bound}')
    plt.xlabel('Number of Samples (n)')
    plt.ylabel('Loss Gap')
    plt.title(f'Sample Complexity - {adjustment or "No Adjustment"}')
    plt.legend()

    # Online Tuning with EXP3
    cum_regrets, losses, best_losses = online_tune_elasticnet_exp3(T, p, lambda_max)
    theoretical_bound = np.sqrt(T) * np.log(T)
    plt.subplot(2, 2, 2)
    plt.plot(range(T), cum_regrets, label='Cumulative Regret')
    plt.plot(range(T), theoretical_bound * np.ones(T), 'r--', label='Õ(√T)')
    plt.xlabel('Time (T)')
    plt.ylabel('Cumulative Regret')
    plt.title('Online Tuning Regret (EXP3)')
    plt.legend()

    # Compute Dispersion
    avg_norm_diff, is_half_dispersed = compute_dispersion(losses, lambda_max)
    print(f"Average Norm Difference: {avg_norm_diff:.4f}, 1/2-Dispersed: {is_half_dispersed}")

    # Loss Surface with Piecewise Structure
    X, y, X_val, y_val = generate_problem(p=p)
    L, G, lambda_range = compute_loss_surface(X, y, X_val, y_val, lambda_max, p)
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(121, projection='3d')
    L1, L2 = np.meshgrid(lambda_range, lambda_range)
    ax.plot_surface(L1, L2, L, cmap='viridis')
    ax.set_xlabel('λ₁')
    ax.set_ylabel('λ₂')
    ax.set_zlabel('Loss')
    ax.set_title('Loss Surface')
    
    ax = fig.add_subplot(122)
    ax.contourf(L1, L2, G, cmap='RdYlGn')
    ax.set_xlabel('λ₁')
    ax.set_ylabel('λ₂')
    ax.set_title('G Regions (Piecewise Decomposition)')
    
    # Multi-Task Learning
    problems = [generate_problem(p=p) for _ in range(3)]  # Reduced from 5 to 3
    _, multi_task_loss = multi_task_elasticnet(problems, 0.1, 0.1, adjustment)
    print(f"Multi-Task Average Loss ({adjustment or 'No Adjustment'}): {multi_task_loss:.4f}")

    plt.tight_layout()
    plt.show()

# --- Streamlit Interface ---
def main():
    st.title("Provably Tuning ElasticNet Across Instances")
    st.write("Explore sample complexity, online regret, dispersion, and piecewise structure for ElasticNet tuning.")
    
    p = st.slider("Feature Dimension (p)", 2, 10, 5)
    T = st.slider("Online Tasks (T)", 10, 200, 100)
    epsilon = st.slider("Error Tolerance (ε)", 0.01, 0.5, 0.1, 0.01)
    delta = st.slider("Failure Probability (δ)", 0.01, 0.1, 0.05, 0.01)
    lambda_max = st.slider("Max λ", 0.5, 2.0, 1.0, 0.1)
    adjustment = st.selectbox("Loss Adjustment", [None, "AIC", "BIC"])
    
    if st.button("Run Simulation"):
        with st.spinner("Running..."):
            progress_bar = st.progress(0)
            plot_results(n=100, p=p, T=T, epsilon=epsilon, delta=delta, lambda_max=lambda_max, adjustment=adjustment)
            progress_bar.progress(100)
        st.success("Simulation complete! Check plots above.")

if __name__ == "__main__":
    # For local testing, run plots directly
    plot_results(n=100, p=5, T=100, epsilon=0.1, delta=0.05, lambda_max=1.0, adjustment="AIC")
    # Uncomment to run Streamlit: `streamlit run this_file.py`
    # main()
