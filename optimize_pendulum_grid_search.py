import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from Digital_twin import DigitalTwin
import time
from scipy.fft import fft
from scipy import signal
import multiprocessing
from functools import partial
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import os
from utility import (
    load_real_data, parallel_cost_function, add_parameter_box, analyze_parameter_sensitivity,
    plot_comprehensive_analysis, simulate_and_plot, generate_optimization_report
)

# Define the CSV filename for data loading
csv_filename = "half_theta_2"

# Grid Search Parameters
I_SCALE_RANGE = np.linspace(0.6, 0.9, 15)  # 15 points from 0.6 to 0.9
DAMPING_RANGE = np.linspace(0.001, 0.04, 15)  # 15 points from 0.001 to 0.04
MASS_RANGE = np.linspace(0.5, 1.5, 15)  # 15 points from 0.5 to 1.5

# Local Optimization Parameters
LOCAL_OPT_METHOD = 'Nelder-Mead'  # Options: 'Nelder-Mead', 'Powell', 'BFGS'
MAX_ITERATIONS = 1000
TOLERANCE = 1e-6

def grid_search_optimize(time_array, theta_real, theta_dot_real, theta0, theta_dot0):
    """Perform grid search optimization"""
    print("Starting Grid Search Optimization...")
    print("=" * 50)
    
    # Initialize variables
    best_cost = float('inf')
    best_params = None
    grid_points = []
    
    # Create results directory if it doesn't exist
    os.makedirs('reports', exist_ok=True)
    
    # Grid search
    total_points = len(I_SCALE_RANGE) * len(DAMPING_RANGE) * len(MASS_RANGE)
    current_point = 0
    
    print(f"Grid Search Parameters:")
    print(f"I_scale range: {I_SCALE_RANGE[0]:.4f} to {I_SCALE_RANGE[-1]:.4f} ({len(I_SCALE_RANGE)} points)")
    print(f"Damping range: {DAMPING_RANGE[0]:.6f} to {DAMPING_RANGE[-1]:.6f} ({len(DAMPING_RANGE)} points)")
    print(f"Mass range: {MASS_RANGE[0]:.4f} to {MASS_RANGE[-1]:.4f} ({len(MASS_RANGE)} points)")
    print(f"Total grid points to evaluate: {total_points}")
    print("-" * 50)
    
    start_time = time.time()
    
    for I_scale in I_SCALE_RANGE:
        for damping in DAMPING_RANGE:
            for mass in MASS_RANGE:
                current_point += 1
                params = [I_scale, damping, mass]
                cost = parallel_cost_function(params, time_array, theta_real, theta_dot_real, theta0, theta_dot0)
                grid_points.append((params, cost))
                
                if cost < best_cost:
                    best_cost = cost
                    best_params = params
                
                # Print progress
                if current_point % 10 == 0 or current_point == total_points:
                    elapsed_time = time.time() - start_time
                    points_per_second = current_point / elapsed_time
                    remaining_points = total_points - current_point
                    estimated_time = remaining_points / points_per_second
                    
                    print("-" * 30)
                    print(f"Progress: {current_point}/{total_points} points ({current_point/total_points*100:.1f}%)")
                    print(f"Current best cost: {best_cost:.6f}")
                    print(f"Estimated time remaining: {estimated_time/60:.1f} minutes")
                    print("-" * 30)
    
    # Sort grid points by cost
    grid_points.sort(key=lambda x: x[1])
    
    # Print best results
    print("\nGrid Search Results:")
    print("-" * 50)
    print(f"Best parameters found:")
    print(f"I_scale: {best_params[0]:.6f}")
    print(f"Damping: {best_params[1]:.6f}")
    print(f"Mass: {best_params[2]:.6f}")
    print(f"Best cost: {best_cost:.6f}")
    
    return best_params, best_cost

def optimize_pendulum_params():
    # Load real data
    time_array, theta_real, theta_dot_real, theta0, theta_dot0 = load_real_data()
    
    # Initialize arrays to track evolution
    best_costs = []
    avg_costs = []
    
    def callback(xk, convergence):
        best_costs.append(convergence)
        avg_costs.append(convergence)  # For grid search, best and avg are the same
    
    # Set up parallel processing
    n_cores = multiprocessing.cpu_count()
    
    # Create a partial function with fixed arguments
    cost_func = partial(parallel_cost_function, 
                       time_array=time_array,
                       theta_real=theta_real,
                       theta_dot_real=theta_dot_real,
                       theta0=theta0,
                       theta_dot0=theta_dot0)
    
    # Run grid search optimization
    best_params, best_cost = grid_search_optimize(time_array, theta_real, theta_dot_real, theta0, theta_dot0)
    
    # Create a result object that matches differential evolution format
    class Result:
        def __init__(self, x, fun, nfev, success, message):
            self.x = x
            self.fun = fun
            self.nfev = nfev
            self.success = success
            self.message = message
    
    result = Result(
        x=best_params,
        fun=best_cost,
        nfev=len(best_costs),
        success=True,
        message="Optimization terminated successfully."
    )
    
    # Plot evolution of costs
    plt.figure(figsize=(10, 6))
    plt.plot(best_costs, label='Best Cost')
    plt.plot(avg_costs, label='Average Cost')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Evolution of Costs During Optimization')
    plt.legend()
    plt.grid(True)
    plt.savefig('reports/GS_evolution.png')
    plt.close()
    
    return result

# Main execution
if __name__ == "__main__":
    # Load real data
    time_array, theta_real, theta_dot_real, theta0, theta_dot0 = load_real_data()
    
    # Run optimization
    result = optimize_pendulum_params()
    
    # Print optimization results
    print("\nGrid Search Results:")
    print("-" * 50)
    print("Best parameters found:")
    print(f"I_scale: {result.x[0]:.6f}")
    print(f"Damping: {result.x[1]:.6f}")
    print(f"Mass: {result.x[2]:.6f}")
    print(f"Best cost: {result.fun:.6f}")
    print(f"Number of evaluations: {result.nfev}")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    
    # Simulate and plot with best parameters
    theta_sim, error, params = simulate_and_plot(result.x, time_array, theta_real, theta_dot_real, theta0, theta_dot0, "half_theta_2", model_name='GS')
    
    # Calculate parameter sensitivities
    sensitivities = analyze_parameter_sensitivity(result, time_array, theta_real, theta_dot_real, theta0, theta_dot0)
    
    # Generate comprehensive optimization report
    generate_optimization_report(
        best_params=result.x,
        time_array=time_array,
        theta_real=theta_real,
        theta_dot_real=theta_dot_real,
        theta0=theta0,
        theta_dot0=theta_dot0,
        title="half_theta_2",
        model_name="GS",
        cost_value=result.fun,
        n_evaluations=result.nfev,
        optimization_time=None,  # Not tracked in this implementation
        sensitivities=sensitivities
    )
