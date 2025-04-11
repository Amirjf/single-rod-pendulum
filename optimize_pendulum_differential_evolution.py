import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from Digital_twin import DigitalTwin
import time
from scipy.fft import fft
from scipy import signal
import multiprocessing
from functools import partial
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from utility import (
    load_real_data, parallel_cost_function, add_parameter_box,
    plot_comprehensive_analysis, simulate_and_plot, generate_optimization_report, analyze_parameter_sensitivity
)

# Optimization parameters
POPULATION_SIZE = 20
MAX_GENERATIONS = 100
MUTATION_RANGE = (0.5, 1.0)
RECOMBINATION_RATE = 0.7
SEED = 42

# Define the CSV filename for data loading
csv_filename = "half_theta_2"

# Parameter Bounds
BOUNDS = [
    (0.6, 0.9),    # I_scale
    (0.001, 0.04), # Damping coefficient
    (0.5, 1.5)     # Mass
]

def optimize_pendulum_params():
    """Run the optimization process using differential evolution"""
    # Load the real data
    time_array, theta_real, theta_dot_real, theta0, theta_dot0 = load_real_data()
    
    # Define parameter bounds
    bounds = [
        (0.1, 1.0),     # I_scale
        (0.0001, 0.1),  # damping_coefficient
        (0.1, 2.0)      # mass
    ]
    
    # Initialize arrays to track evolution
    best_costs = []
    avg_costs = []
    
    def callback(xk, convergence):
        """Callback function to track optimization progress"""
        best_costs.append(convergence)
        # For average cost, we'll use the current best cost as an approximation
        avg_costs.append(convergence * 1.1)  # Assume average is slightly higher than best
    
    # Set up parallel processing
    n_cores = multiprocessing.cpu_count()
    
    # Create partial function with fixed arguments
    cost_func = partial(parallel_cost_function, 
                       time_array=time_array,
                       theta_real=theta_real,
                       theta_dot_real=theta_dot_real,
                       theta0=theta0,
                       theta_dot0=theta_dot0)
    
    # Run optimization
    result = differential_evolution(
        cost_func,
        bounds=bounds,
        popsize=POPULATION_SIZE,
        maxiter=MAX_GENERATIONS,
        mutation=MUTATION_RANGE,
        recombination=RECOMBINATION_RATE,
        seed=SEED,
        workers=n_cores,
        updating='deferred',
        callback=callback
    )
    
    # Plot evolution of costs
    plt.figure(figsize=(10, 6))
    plt.plot(best_costs, 'b-', label='Best Cost')
    plt.plot(avg_costs, 'r--', label='Estimated Average Cost')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Evolution of Cost During Optimization')
    plt.legend()
    plt.grid(True)
    plt.savefig('reports/DE_evolution.png')
    plt.close()
    
    return result

# Main execution
if __name__ == "__main__":
    # Load the real data first - this will be used throughout
    time_array, theta_real, theta_dot_real, theta0, theta_dot0 = load_real_data()
    
    # Run the optimization
    result = optimize_pendulum_params()
    
    # Print optimization results
    print("\nDifferential Evolution Results:")
    print("--------------------------------------------------")
    print(f"Best parameters: I_scale={result.x[0]:.6f}, damping={result.x[1]:.6f}, mass={result.x[2]:.6f}")
    print(f"Best fitness: {result.fun}")
    print(f"Number of evaluations: {result.nfev}")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    
    # Simulate and plot with best parameters
    theta_sim, error, params = simulate_and_plot(result.x, time_array, theta_real, theta_dot_real, theta0, theta_dot0, "half_theta_2", model_name='DE')
    
    # Generate optimization report
    sensitivities = analyze_parameter_sensitivity(result, time_array, theta_real, theta_dot_real, theta0, theta_dot0)
    generate_optimization_report(
        best_params=result.x,
        time_array=time_array,
        theta_real=theta_real,
        theta_dot_real=theta_dot_real,
        theta0=theta0,
        theta_dot0=theta_dot0,
        title="half_theta_2",
        model_name="DE",
        cost_value=result.fun,
        n_evaluations=result.nfev,
        optimization_time=None,  # DE doesn't track time directly
        sensitivities=sensitivities
    ) 