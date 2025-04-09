"""
Grid Search and Local Optimization for Pendulum Parameter Tuning
This script implements a systematic grid search followed by local optimization
to find optimal parameters for the pendulum model.
"""
from Optimized_digital_twin import analyze_parameter_sensitivity, optimize_pendulum_params,load_real_data, print_optimization_results, save_optimization_report, simulate_and_plot



# Define the CSV filename for data loading
csv_filename = "half_theta_2"

# Main execution
if __name__ == "__main__":
    # Load the real data first - this will be used throughout
    time_array, theta_real, theta_dot_real, theta0, theta_dot0 = load_real_data()
    
    # Run the optimization
    result = optimize_pendulum_params()
    
    # Get the best parameters
    I_scale, damping_coefficient, mass = result.x
    
    # Simulate and plot with best parameters
    theta_sim, error, I_scale, damping_coefficient, mass = simulate_and_plot(result.x)
    
    # Print comprehensive results
    print_optimization_results(result, error, time_array, theta_real, theta_sim)
    
    # Save optimization report
    save_optimization_report(result, error, time_array, theta_real, theta_sim)
    
    # Analyze parameter sensitivity
    analyze_parameter_sensitivity(result, time_array, theta_real, theta_dot_real, theta0, theta_dot0)