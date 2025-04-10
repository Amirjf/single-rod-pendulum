import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy import signal
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from Digital_twin import DigitalTwin
import time
from utility import (
    load_real_data, create_twin_with_params, calculate_fft_analysis,
    calculate_velocity, calculate_errors, find_peaks_and_zeros,
    plot_comprehensive_analysis, simulate_and_plot
)

def compare_optimization_results():
    # Load real data
    time_array, theta_real, theta_dot_real, theta0, theta_dot0 = load_real_data()
    
    # Best parameters from each optimization method
    de_params = [0.705700, 0.008006, 0.527869]  # Differential Evolution
    ga_params = [0.711055, 0.025936, 1.543882]  # Genetic Algorithm
    gs_params = [0.707143, 0.012143, 0.785714]  # Grid Search
    
    # Simulate with each set of parameters
    theta_sim_de, error_de, _ = simulate_and_plot(de_params, time_array, theta_real, theta_dot_real, 
                                                theta0, theta_dot0, "half_theta_2", "DE")
    theta_sim_ga, error_ga, _ = simulate_and_plot(ga_params, time_array, theta_real, theta_dot_real, 
                                                theta0, theta_dot0, "half_theta_2", "GA")
    theta_sim_gs, error_gs, _ = simulate_and_plot(gs_params, time_array, theta_real, theta_dot_real, 
                                                theta0, theta_dot0, "half_theta_2", "GS")
    
    # Create comparison plots
    plt.figure(figsize=(20, 15))
    
    # 1. Time domain comparison
    plt.subplot(3, 2, 1)
    plt.plot(time_array, theta_real, 'k-', label='Real', alpha=0.7)
    plt.plot(time_array, theta_sim_de, 'r--', label='DE', alpha=0.7)
    plt.plot(time_array, theta_sim_ga, 'b--', label='GA', alpha=0.7)
    plt.plot(time_array, theta_sim_gs, 'g--', label='GS', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.title('Time Domain Comparison')
    plt.grid(True)
    plt.legend()
    
    # 2. Frequency domain comparison
    dt = time_array[1] - time_array[0]
    real_mag_norm, freq = calculate_fft_analysis(theta_real, dt)
    de_mag_norm, _ = calculate_fft_analysis(theta_sim_de, dt)
    ga_mag_norm, _ = calculate_fft_analysis(theta_sim_ga, dt)
    gs_mag_norm, _ = calculate_fft_analysis(theta_sim_gs, dt)
    
    plt.subplot(3, 2, 2)
    plt.plot(freq, real_mag_norm, 'k-', label='Real', alpha=0.7)
    plt.plot(freq, de_mag_norm, 'r--', label='DE', alpha=0.7)
    plt.plot(freq, ga_mag_norm, 'b--', label='GA', alpha=0.7)
    plt.plot(freq, gs_mag_norm, 'g--', label='GS', alpha=0.7)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized Magnitude')
    plt.title('Frequency Domain Comparison')
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 2)
    
    # 3. Phase space plots
    theta_dot_real = calculate_velocity(theta_real, dt)
    theta_dot_de = calculate_velocity(theta_sim_de, dt)
    theta_dot_ga = calculate_velocity(theta_sim_ga, dt)
    theta_dot_gs = calculate_velocity(theta_sim_gs, dt)
    
    plt.subplot(3, 2, 3)
    plt.plot(theta_real, theta_dot_real, 'k-', label='Real', alpha=0.7)
    plt.plot(theta_sim_de, theta_dot_de, 'r--', label='DE', alpha=0.7)
    plt.plot(theta_sim_ga, theta_dot_ga, 'b--', label='GA', alpha=0.7)
    plt.plot(theta_sim_gs, theta_dot_gs, 'g--', label='GS', alpha=0.7)
    plt.xlabel('Angle (rad)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.title('Phase Space Comparison')
    plt.grid(True)
    plt.legend()
    
    # 4. Error distributions
    plt.subplot(3, 2, 4)
    plt.hist(error_de, bins=50, alpha=0.3, label='DE', color='r')
    plt.hist(error_ga, bins=50, alpha=0.3, label='GA', color='b')
    plt.hist(error_gs, bins=50, alpha=0.3, label='GS', color='g')
    plt.xlabel('Error (rad)')
    plt.ylabel('Count')
    plt.title('Error Distribution Comparison')
    plt.grid(True)
    plt.legend()
    
    # 5. Parameter comparison
    params = np.array([de_params, ga_params, gs_params])
    param_names = ['I_scale', 'Damping', 'Mass']
    methods = ['DE', 'GA', 'GS']
    
    plt.subplot(3, 2, 5)
    x = np.arange(len(param_names))
    width = 0.25
    
    plt.bar(x - width, params[0], width, label='DE')
    plt.bar(x, params[1], width, label='GA')
    plt.bar(x + width, params[2], width, label='GS')
    
    plt.xlabel('Parameters')
    plt.ylabel('Value')
    plt.title('Parameter Comparison')
    plt.xticks(x, param_names)
    plt.grid(True)
    plt.legend()
    
    # 6. Performance metrics
    metrics = {
        'DE': {'RMS': np.sqrt(np.mean(error_de**2)), 'Max': np.max(np.abs(error_de))},
        'GA': {'RMS': np.sqrt(np.mean(error_ga**2)), 'Max': np.max(np.abs(error_ga))},
        'GS': {'RMS': np.sqrt(np.mean(error_gs**2)), 'Max': np.max(np.abs(error_gs))}
    }
    
    plt.subplot(3, 2, 6)
    x = np.arange(len(methods))
    width = 0.35
    
    plt.bar(x - width/2, [m['RMS'] for m in metrics.values()], width, label='RMS Error')
    plt.bar(x + width/2, [m['Max'] for m in metrics.values()], width, label='Max Error')
    
    plt.xlabel('Optimization Method')
    plt.ylabel('Error (rad)')
    plt.title('Performance Metrics Comparison')
    plt.xticks(x, methods)
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('reports/optimization_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    compare_optimization_results() 