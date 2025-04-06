"""
Grid Search and Local Optimization for Pendulum Parameter Tuning
This script implements a systematic grid search followed by local optimization
to find optimal parameters for the pendulum model.
"""

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
import os

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

class ModifiedDigitalTwin(DigitalTwin):
    """Extended DigitalTwin with simplified damping model for pendulum optimization"""
    
    def __init__(self):
        super().__init__()
        # Physical parameters
        self.g = 9.81  # gravity (m/s²)
        self.l = 0.35  # length (m)
        self.mp = 0.35  # mass (kg)
        self.I_scale = 0.35  # moment of inertia scale
        self.damping_coefficient = 0.0  # damping coefficient
        
        # Derived parameters
        self.I = self.I_scale * self.mp * self.l**2  # moment of inertia
    
    def get_theta_double_dot(self, theta, theta_dot):
        """Calculate angular acceleration based on current state"""
        # Gravity term
        torque_gravity = -(self.g / (self.I_scale * self.l)) * np.sin(theta)
        
        # Damping term
        torque_damping = -(self.damping_coefficient / self.I) * theta_dot
        
        return torque_gravity + torque_damping

    def simulate_passive(self, theta0, theta_dot0, time_array):
        """Simulate passive pendulum motion with enhanced static friction"""
        self.theta = theta0
        self.theta_dot = theta_dot0
        theta_history = [self.theta]
        
        static_threshold = 0.5  # Velocity threshold for enhanced friction
        
        for i in range(1, len(time_array)):
            dt = time_array[i] - time_array[i - 1]
            
            # Calculate acceleration
            theta_ddot = self.get_theta_double_dot(self.theta, self.theta_dot)
            
            # Apply smooth enhanced friction near zero velocity
            enhancement = 5.0 * (1.0 - np.tanh(abs(self.theta_dot) / static_threshold)**2)
            theta_ddot -= enhancement * (self.damping_coefficient / (self.mp * self.I)) * self.theta_dot
            
            # Update state
            self.theta_dot += theta_ddot * dt
            self.theta += self.theta_dot * dt
            theta_history.append(self.theta)
        
        return np.array(theta_history)

# Data loading function
def load_real_data(filename=None, start_time=1.0):
    """Load and preprocess real pendulum data from CSV file"""
    if filename is None:
        filename = f"datasets/filtered_datasets/{csv_filename}_kalman_output.csv"
    
    # Load and filter data
    df = pd.read_csv(filename)
    df_trimmed = df[df['time_sec'] >= start_time].reset_index(drop=True)
    
    # Extract time and angle data
    time_array = df_trimmed['time_sec'].values
    theta_real = df_trimmed['theta_kalman'].values
    theta_dot_real = df_trimmed['theta_dot_kalman'].values
    
    # Get initial conditions
    theta0 = theta_real[0]
    theta_dot0 = theta_dot_real[0]
    
    print(f"Initial conditions: theta0 = {theta0:.4f}, theta_dot0 = {theta_dot0:.4f}")
    return time_array, theta_real, theta_dot_real, theta0, theta_dot0

def parallel_cost_function(params, time_array, theta_real, theta_dot_real, theta0, theta_dot0):
    """Calculate the cost function for a given set of parameters"""
    I_scale, damping_coefficient, mass = params
    
    # Create twin with parameters
    twin = ModifiedDigitalTwin()
    twin.mp = mass
    twin.l = 0.35
    twin.I_scale = I_scale
    twin.I = I_scale * twin.mp * twin.l**2
    twin.damping_coefficient = damping_coefficient
    twin.c_air = 0.0
    twin.currentmotor_acceleration = 0
    
    # Simulate
    theta_sim = twin.simulate_passive(theta0, theta_dot0, time_array)
    
    # Calculate velocity from position data
    dt = time_array[1] - time_array[0]
    theta_dot_sim = np.gradient(theta_sim, dt)
    
    # Calculate errors
    min_len = min(len(theta_sim), len(theta_real))
    error = theta_sim[:min_len] - theta_real[:min_len]
    
    # Time domain error with exponential weighting
    max_amplitude = np.max(np.abs(theta_real[:min_len]))
    time_weights = np.exp(np.linspace(0, 1, min_len))
    time_domain_error = np.mean(time_weights * (error/max_amplitude)**2)
    
    # Frequency domain error
    n_points = 8192
    real_fft = fft(theta_real[:min_len], n_points)
    sim_fft = fft(theta_sim[:min_len], n_points)
    real_mag = np.abs(real_fft[:n_points//2])
    sim_mag = np.abs(sim_fft[:n_points//2])
    real_mag_norm = real_mag / np.max(real_mag)
    sim_mag_norm = sim_mag / np.max(sim_mag)
    freq = np.fft.fftfreq(n_points, d=dt)[:n_points//2]
    freq_mask = (freq >= 0.9) & (freq <= 1.1)
    freq_error = np.mean((real_mag_norm[freq_mask] - sim_mag_norm[freq_mask])**2)
    
    # Amplitude error
    real_peaks = signal.find_peaks(np.abs(theta_real[:min_len]), distance=15)[0]
    sim_peaks = signal.find_peaks(np.abs(theta_sim[:min_len]), distance=15)[0]
    n_peaks = min(len(real_peaks), len(sim_peaks))
    real_amplitudes = np.abs(theta_real[real_peaks[:n_peaks]]) / max_amplitude
    sim_amplitudes = np.abs(theta_sim[sim_peaks[:n_peaks]]) / max_amplitude
    amplitude_error = np.mean((real_amplitudes - sim_amplitudes)**2)
    
    # Decay error
    real_decay = real_amplitudes[1:] / real_amplitudes[:-1]
    sim_decay = sim_amplitudes[1:] / sim_amplitudes[:-1]
    decay_error = np.mean((real_decay - sim_decay)**2)
    
    # Energy conservation error
    l = twin.l
    g = twin.g
    E_real = 0.5 * l**2 * theta_dot_real[:min_len]**2 + g * l * (1 - np.cos(theta_real[:min_len]))
    E_sim = 0.5 * l**2 * theta_dot_sim**2 + g * l * (1 - np.cos(theta_sim[:min_len]))
    energy_error = np.mean((E_sim - E_real)/np.max(E_real))**2
    
    # Total cost with weights
    total_cost = (50 * time_domain_error + 
                 600 * freq_error + 
                 200 * amplitude_error + 
                 200 * decay_error + 
                 energy_error)
    
    return total_cost

def grid_search_optimize():
    """Perform grid search optimization"""
    print("Starting Grid Search Optimization...")
    print("=" * 50)
    
    # Load real data
    time_array, theta_real, theta_dot_real, theta0, theta_dot0 = load_real_data()
    
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
                    
                    print(f"Progress: {current_point}/{total_points} points "
                          f"({current_point/total_points*100:.1f}%)")
                    print(f"Current best cost: {best_cost:.6f}")
                    print(f"Estimated time remaining: {estimated_time/60:.1f} minutes")
                    print("-" * 30)
    
    # Sort grid points by cost
    grid_points.sort(key=lambda x: x[1])
    
    # Save grid search results
    with open('reports/grid_search_results.txt', 'w') as f:
        f.write("Grid Search Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total points evaluated: {total_points}\n")
        f.write(f"Best parameters found:\n")
        f.write(f"I_scale: {best_params[0]:.6f}\n")
        f.write(f"Damping: {best_params[1]:.8f}\n")
        f.write(f"Mass: {best_params[2]:.6f}\n")
        f.write(f"Best cost: {best_cost:.6f}\n\n")
        f.write("Top 10 parameter combinations:\n")
        f.write("-" * 50 + "\n")
        for i, (params, cost) in enumerate(grid_points[:10], 1):
            f.write(f"{i}. I_scale={params[0]:.6f}, Damping={params[1]:.8f}, "
                   f"Mass={params[2]:.6f}, Cost={cost:.6f}\n")
    
    print("\nGrid Search completed!")
    print(f"Best parameters found: I_scale={best_params[0]:.6f}, "
          f"Damping={best_params[1]:.8f}, Mass={best_params[2]:.6f}")
    print(f"Best cost: {best_cost:.6f}")
    
    return best_params, best_cost

def local_optimization(initial_params):
    """Perform local optimization starting from the best grid search point"""
    print("\nStarting Local Optimization...")
    print("=" * 50)
    
    # Load real data
    time_array, theta_real, theta_dot_real, theta0, theta_dot0 = load_real_data()
    
    # Define bounds for parameters
    bounds = [
        (initial_params[0] * 0.9, initial_params[0] * 1.1),  # I_scale ±10%
        (initial_params[1] * 0.9, initial_params[1] * 1.1),  # Damping ±10%
        (initial_params[2] * 0.9, initial_params[2] * 1.1)   # Mass ±10%
    ]
    
    # Define the objective function
    def objective(params):
        return parallel_cost_function(params, time_array, theta_real, theta_dot_real, theta0, theta_dot0)
    
    # Run local optimization
    result = minimize(
        objective,
        x0=initial_params,
        method=LOCAL_OPT_METHOD,
        bounds=bounds,
        options={
            'maxiter': MAX_ITERATIONS,
            'xatol': TOLERANCE,
            'fatol': TOLERANCE
        }
    )
    
    print(f"\nLocal Optimization Results ({LOCAL_OPT_METHOD}):")
    print(f"Success: {result.success}")
    print(f"Number of iterations: {result.nit}")
    print(f"Number of function evaluations: {result.nfev}")
    print(f"Final cost value: {result.fun:.6f}")
    print(f"Optimal parameters:")
    print(f"I_scale: {result.x[0]:.6f}")
    print(f"Damping: {result.x[1]:.8f}")
    print(f"Mass: {result.x[2]:.6f}")
    
    return result

def optimize_pendulum_params():
    """Main optimization function combining grid search and local optimization"""
    # First perform grid search
    best_grid_params, best_grid_cost = grid_search_optimize()
    
    # Then perform local optimization starting from the best grid point
    result = local_optimization(best_grid_params)
    
    # Create a result object similar to scipy.optimize.differential_evolution
    class OptimizationResult:
        def __init__(self, x, fun, success, nfev, nit, message=None):
            self.x = x
            self.fun = fun
            self.success = success
            self.nfev = nfev
            self.nit = nit
            self.message = message
    
    # Combine results from both methods
    final_result = OptimizationResult(
        x=result.x,
        fun=result.fun,
        success=result.success,
        nfev=result.nfev,
        nit=result.nit,
        message=f"Grid Search + Local Optimization ({LOCAL_OPT_METHOD})"
    )
    
    return final_result

def add_parameter_box(fig, params, twin, rms_error, max_error):
    """Add parameter information box to the figure"""
    I_scale, damping_coefficient, mass = params
    param_text = (
        f'Model Parameters:\n'
        f'I_scale = {I_scale:.6f}\n'
        f'damping_coefficient = {damping_coefficient:.6f}\n'
        f'mass = {mass:.6f} kg\n'
        f'\nPhysical Values:\n'
        f'Length = {twin.l:.3f} m\n'
        f'I_eff = {twin.I:.6f} kg⋅m²\n'
        f'\nPerformance:\n'
        f'RMS Error = {rms_error:.6f} rad\n'
        f'Max Error = {max_error:.6f} rad'
    )
    fig.text(0.98, 0.98, param_text, fontsize=10, family='monospace', 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.8))

def plot_comprehensive_analysis(theta_real, theta_sim, time_array, theta_dot_real=None):
    """Create comprehensive analysis plots with multiple metrics"""
    min_len = min(len(theta_sim), len(theta_real))
    dt = time_array[1] - time_array[0]
    
    # Create twin to get g and l values
    twin = ModifiedDigitalTwin()
    
    # Calculate theta_dot for simulation
    theta_dot_sim = np.gradient(theta_sim, dt)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 20))
    
    # 1. Time-domain position comparison
    ax1 = fig.add_subplot(4, 2, 1)
    ax1.plot(time_array, theta_real, 'b-', label='Real θ')
    ax1.plot(time_array, theta_sim, 'r--', label='Simulated θ')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angle (rad)')
    ax1.set_title('Pendulum Angle: Real vs Simulated')
    ax1.grid(True)
    ax1.legend()
    
    # 2. Velocity comparison
    ax2 = fig.add_subplot(4, 2, 2)
    ax2.plot(time_array[:min_len], theta_dot_real[:min_len], 'b-', label='Real θ̇')
    ax2.plot(time_array[:min_len], theta_dot_sim[:min_len], 'r--', label='Simulated θ̇')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angular Velocity (rad/s)')
    ax2.set_title('Angular Velocity Comparison')
    ax2.grid(True)
    ax2.legend()
    
    # 3. Phase space plot
    ax3 = fig.add_subplot(4, 2, 3)
    ax3.plot(theta_real[:min_len], theta_dot_real[:min_len], 'b-', label='Real', alpha=0.5)
    ax3.plot(theta_sim[:min_len], theta_dot_sim[:min_len], 'r--', label='Simulated', alpha=0.5)
    ax3.set_xlabel('θ (rad)')
    ax3.set_ylabel('θ̇ (rad/s)')
    ax3.set_title('Phase Space Plot')
    ax3.grid(True)
    ax3.legend()
    
    # 4. Zero-crossing and Peak timing
    real_zeros = np.where(np.diff(np.signbit(theta_real[:min_len])))[0]
    sim_zeros = np.where(np.diff(np.signbit(theta_sim[:min_len])))[0]
    
    real_peaks, _ = signal.find_peaks(np.abs(theta_real[:min_len]))
    sim_peaks, _ = signal.find_peaks(np.abs(theta_sim[:min_len]))
    
    ax4 = fig.add_subplot(4, 2, 4)
    ax4.plot(time_array[:min_len], theta_real[:min_len], 'b-', alpha=0.5)
    ax4.plot(time_array[:min_len], theta_sim[:min_len], 'r--', alpha=0.5)
    ax4.plot(time_array[real_zeros], np.zeros_like(real_zeros), 'bo', label='Real Zeros')
    ax4.plot(time_array[sim_zeros], np.zeros_like(sim_zeros), 'ro', label='Sim Zeros')
    ax4.plot(time_array[real_peaks], theta_real[real_peaks], 'bx', label='Real Peaks')
    ax4.plot(time_array[sim_peaks], theta_sim[sim_peaks], 'rx', label='Sim Peaks')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Angle (rad)')
    ax4.set_title('Zero Crossings and Peaks')
    ax4.grid(True)
    ax4.legend()
    
    # 5. Local frequency variation
    from scipy.signal import hilbert
    analytic_real = hilbert(theta_real[:min_len])
    analytic_sim = hilbert(theta_sim[:min_len])
    
    inst_freq_real = np.diff(np.unwrap(np.angle(analytic_real))) / (2.0*np.pi*dt)
    inst_freq_sim = np.diff(np.unwrap(np.angle(analytic_sim))) / (2.0*np.pi*dt)
    
    ax5 = fig.add_subplot(4, 2, 5)
    ax5.plot(time_array[1:min_len], inst_freq_real, 'b-', label='Real')
    ax5.plot(time_array[1:min_len], inst_freq_sim, 'r--', label='Simulated')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Frequency (Hz)')
    ax5.set_title('Instantaneous Frequency')
    ax5.grid(True)
    ax5.legend()
    
    # 6. Error distributions
    position_error = theta_sim[:min_len] - theta_real[:min_len]
    velocity_error = theta_dot_sim[:min_len] - theta_dot_real[:min_len]
    
    ax6 = fig.add_subplot(4, 2, 6)
    ax6.hist(position_error, bins=50, alpha=0.5, label='Position Error')
    ax6.hist(velocity_error, bins=50, alpha=0.5, label='Velocity Error')
    ax6.set_xlabel('Error')
    ax6.set_ylabel('Count')
    ax6.set_title('Error Distributions')
    ax6.grid(True)
    ax6.legend()
    
    # 7. Frequency domain analysis
    ax7 = fig.add_subplot(4, 2, 7)
    n_points = 8192
    freq = np.fft.fftfreq(n_points, d=dt)[:n_points//2]
    real_fft = np.abs(fft(theta_real[:min_len], n_points)[:n_points//2])
    sim_fft = np.abs(fft(theta_sim[:min_len], n_points)[:n_points//2])
    ax7.plot(freq, real_fft, 'b-', label='Real Data')
    ax7.plot(freq, sim_fft, 'r--', label='Simulation')
    ax7.set_xlabel('Frequency (Hz)')
    ax7.set_ylabel('Magnitude')
    ax7.set_title('Frequency Domain Analysis')
    ax7.set_xlim(0, 2)
    ax7.grid(True)
    ax7.legend()
    
    plt.tight_layout()
    return fig

def simulate_and_plot(params):
    """Simulate pendulum motion and create analysis plots"""
    # Load real data
    time_array, theta_real, theta_dot_real, theta0, theta_dot0 = load_real_data()
    
    I_scale, damping_coefficient, mass = params
    
    # Create twin with parameters
    twin = ModifiedDigitalTwin()
    twin.mp = mass
    twin.l = 0.35
    twin.I_scale = I_scale
    twin.I = I_scale * twin.mp * twin.l**2
    twin.damping_coefficient = damping_coefficient
    twin.c_air = 0.0
    twin.currentmotor_acceleration = 0
    
    # Simulate
    theta_sim = twin.simulate_passive(theta0, theta_dot0, time_array)
    
    # Calculate velocity from position data
    dt = time_array[1] - time_array[0]
    theta_dot_sim = np.gradient(theta_sim, dt)
    theta_dot_real = np.gradient(theta_real, dt)
    
    # Create plots
    fig = plot_comprehensive_analysis(theta_real, theta_sim, time_array, theta_dot_real)
    
    # Calculate metrics
    min_len = min(len(theta_sim), len(theta_real))
    error = theta_sim[:min_len] - theta_real[:min_len]
    rms_error = np.sqrt(np.mean(error**2))
    max_error = np.max(np.abs(error))
    
    # Add parameter box
    add_parameter_box(fig, params, twin, rms_error, max_error)
    
    # Save plot in reports folder with GS-specific naming
    plt.savefig(f'reports/half_theta_2_pendulum_GS_analysis.png', dpi=300)
    plt.show()
    
    return theta_sim, error, I_scale, damping_coefficient, mass

def analyze_parameter_sensitivity(result, time_array, theta_real, theta_dot_real, theta0, theta_dot0):
    """Analyze sensitivity of cost function to parameter variations"""
    # Get optimal parameters
    optimal_I_scale = result.x[0]
    optimal_damping = result.x[1]
    optimal_mass = result.x[2]
    
    # Define variation percentages
    variations = [-0.1, -0.05, 0.0, 0.05, 0.1]  # -10%, -5%, 0%, +5%, +10%
    
    # Initialize arrays for results
    I_costs = []
    mass_costs = []
    damping_costs = []
    
    # Test I_scale variations
    for var in variations:
        test_I = optimal_I_scale * (1 + var)
        cost = parallel_cost_function(
            [test_I, optimal_damping, optimal_mass],
            time_array, theta_real, theta_dot_real, theta0, theta_dot0
        )
        I_costs.append(cost)
    
    # Test mass variations
    for var in variations:
        test_mass = optimal_mass * (1 + var)
        cost = parallel_cost_function(
            [optimal_I_scale, optimal_damping, test_mass],
            time_array, theta_real, theta_dot_real, theta0, theta_dot0
        )
        mass_costs.append(cost)
    
    # Test damping coefficient variations
    for var in variations:
        test_damping = optimal_damping * (1 + var)
        cost = parallel_cost_function(
            [optimal_I_scale, test_damping, optimal_mass],
            time_array, theta_real, theta_dot_real, theta0, theta_dot0
        )
        damping_costs.append(cost)
    
    # Create sensitivity plot
    plt.figure(figsize=(12, 6))
    
    # Plot I_scale sensitivity
    plt.subplot(1, 3, 1)
    plt.plot([v*100 for v in variations], I_costs, 'b-o')
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('I_scale Variation (%)')
    plt.ylabel('Cost Value')
    plt.title('I_scale Sensitivity')
    plt.grid(True)
    
    # Plot mass sensitivity
    plt.subplot(1, 3, 2)
    plt.plot([v*100 for v in variations], mass_costs, 'r-o')
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Mass Variation (%)')
    plt.ylabel('Cost Value')
    plt.title('Mass Sensitivity')
    plt.grid(True)
    
    # Plot damping coefficient sensitivity
    plt.subplot(1, 3, 3)
    plt.plot([v*100 for v in variations], damping_costs, 'g-o')
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Damping Coefficient Variation (%)')
    plt.ylabel('Cost Value')
    plt.title('Damping Coefficient Sensitivity')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('reports/GS_parameter_sensitivity.png')
    plt.close()
    
    # Print sensitivity analysis results
    print("\nParameter Sensitivity Analysis:")
    print("I_scale variations:")
    for var, cost in zip(variations, I_costs):
        print(f"  {var*100:>6.1f}%: {cost:>10.4f}")
    
    print("\nMass variations:")
    for var, cost in zip(variations, mass_costs):
        print(f"  {var*100:>6.1f}%: {cost:>10.4f}")
    
    print("\nDamping coefficient variations:")
    for var, cost in zip(variations, damping_costs):
        print(f"  {var*100:>6.1f}%: {cost:>10.4f}")
    
    # Calculate and print sensitivity metrics
    def calculate_sensitivity(costs):
        base_cost = costs[2]  # Cost at 0% variation
        max_increase = max(costs) - base_cost
        max_decrease = base_cost - min(costs)
        return max_increase/base_cost, max_decrease/base_cost
    
    I_sensitivity = calculate_sensitivity(I_costs)
    mass_sensitivity = calculate_sensitivity(mass_costs)
    damping_sensitivity = calculate_sensitivity(damping_costs)
    
    print("\nSensitivity Metrics (max increase/decrease relative to base cost):")
    print(f"I_scale:     +{I_sensitivity[0]*100:.1f}% / -{I_sensitivity[1]*100:.1f}%")
    print(f"Mass:        +{mass_sensitivity[0]*100:.1f}% / -{mass_sensitivity[1]*100:.1f}%")
    print(f"Damping:     +{damping_sensitivity[0]*100:.1f}% / -{damping_sensitivity[1]*100:.1f}%")

def print_optimization_results(result, error, time_array, theta_real, theta_sim):
    """Print comprehensive optimization results with physical parameters"""
    # Create twin to get g and l values
    twin = ModifiedDigitalTwin()
    
    # Unpack parameters
    I_scale, damping_coefficient, mass = result.x
    l = twin.l  # pendulum length from class
    g = twin.g  # gravity from class
    I = I_scale * mass * l**2  # moment of inertia
    
    # Calculate natural frequency
    omega_n = np.sqrt(g/l)  # natural frequency in rad/s
    f_n = omega_n/(2*np.pi)  # natural frequency in Hz
    
    # Calculate damping ratio
    zeta = damping_coefficient/(2*np.sqrt(I*g/l))  # damping ratio
    
    # Calculate quality factor
    Q = 1/(2*zeta) if zeta > 0 else float('inf')
    
    # Calculate time constant
    tau = 1/(zeta*omega_n) if zeta > 0 else float('inf')
    
    # Calculate cost function breakdown
    min_len = min(len(theta_sim), len(theta_real))
    dt = time_array[1] - time_array[0]
    theta_dot_sim = np.gradient(theta_sim[:min_len], dt)
    
    # Time domain error
    max_amplitude = np.max(np.abs(theta_real[:min_len]))
    time_weights = np.exp(np.linspace(0, 1, min_len))
    time_domain_error = np.mean(time_weights * ((theta_sim[:min_len] - theta_real[:min_len])/max_amplitude)**2)
    
    # Frequency error
    n_points = 8192
    real_fft = fft(theta_real[:min_len], n_points)
    sim_fft = fft(theta_sim[:min_len], n_points)
    real_mag = np.abs(real_fft[:n_points//2])
    sim_mag = np.abs(sim_fft[:n_points//2])
    real_mag_norm = real_mag / np.max(real_mag)
    sim_mag_norm = sim_mag / np.max(sim_mag)
    freq = np.fft.fftfreq(n_points, d=dt)[:n_points//2]
    freq_mask = (freq >= 0.9) & (freq <= 1.1)
    freq_error = np.mean((real_mag_norm[freq_mask] - sim_mag_norm[freq_mask])**2)
    
    # Amplitude error
    real_peaks = signal.find_peaks(np.abs(theta_real[:min_len]), distance=15)[0]
    sim_peaks = signal.find_peaks(np.abs(theta_sim[:min_len]), distance=15)[0]
    n_peaks = min(len(real_peaks), len(sim_peaks))
    real_amplitudes = np.abs(theta_real[real_peaks[:n_peaks]]) / max_amplitude
    sim_amplitudes = np.abs(theta_sim[sim_peaks[:n_peaks]]) / max_amplitude
    amplitude_error = np.mean((real_amplitudes - sim_amplitudes)**2)
    
    # Decay error
    real_decay = real_amplitudes[1:] / real_amplitudes[:-1]
    sim_decay = sim_amplitudes[1:] / sim_amplitudes[:-1]
    decay_error = np.mean((real_decay - sim_decay)**2)
    
    # Statistical Analysis
    error_mean = np.mean(error)
    error_std = np.std(error)
    error_skew = stats.skew(error)
    error_kurtosis = stats.kurtosis(error)
    
    # Physical Validation
    # Calculate measured natural frequency from zero crossings
    zero_crossings = np.where(np.diff(np.signbit(theta_real[:min_len])))[0]
    if len(zero_crossings) >= 2:
        measured_period = 2 * (time_array[zero_crossings[-1]] - time_array[zero_crossings[0]]) / (len(zero_crossings) - 1)
        measured_freq = 1/measured_period
        freq_error_percent = abs(measured_freq - f_n)/f_n * 100
    else:
        measured_freq = f_n
        freq_error_percent = 0
    
    # Energy conservation analysis
    E_real = 0.5 * l**2 * theta_dot_real[:min_len]**2 + g * l * (1 - np.cos(theta_real[:min_len]))
    E_sim = 0.5 * l**2 * theta_dot_sim**2 + g * l * (1 - np.cos(theta_sim[:min_len]))
    energy_error = np.mean((E_sim - E_real)/np.max(E_real))**2
    
    # Print results
    print("\nOptimization Results:")
    print("-" * 50)
    print("Physical Parameters:")
    print(f"Mass (mp): {mass:.4f} kg")
    print(f"Length (l): {l:.4f} m")
    print(f"Moment of Inertia Scale: {I_scale:.4f}")
    print(f"Effective I: {I:.6f} kg⋅m²")
    print(f"Damping Coefficient: {damping_coefficient:.8f}")
    
    print("\nDynamic Characteristics:")
    print(f"Natural Frequency: {f_n:.4f} Hz")
    print(f"Measured Frequency: {measured_freq:.4f} Hz")
    print(f"Frequency Error: {freq_error_percent:.2f}%")
    print(f"Damping Ratio: {zeta:.4f}")
    print(f"Quality Factor: {Q:.2f}")
    print(f"Time Constant: {tau:.4f} s")
    
    print("\nPerformance Metrics:")
    print(f"RMS Error: {np.sqrt(np.mean(error**2)):.4f} rad")
    print(f"Max Absolute Error: {np.max(np.abs(error)):.4f} rad")
    print(f"Mean Error: {error_mean:.4f} rad")
    print(f"Error Std Dev: {error_std:.4f} rad")
    print(f"Error Skewness: {error_skew:.4f}")
    print(f"Error Kurtosis: {error_kurtosis:.4f}")
    
    print("\nCost Function Breakdown:")
    print(f"Time Domain Error: {50*time_domain_error:.4f}")
    print(f"Frequency Error: {600*freq_error:.4f}")
    print(f"Amplitude Error: {200*amplitude_error:.4f}")
    print(f"Decay Error: {200*decay_error:.4f}")
    print(f"Energy Error: {energy_error:.4f}")
    print(f"Total Cost: {result.fun:.4f}")
    
    print("\nOptimization Summary:")
    print(f"Success: {result.success}")
    print(f"Number of evaluations: {result.nfev}")
    print(f"Number of iterations: {result.nit}")
    print(f"Final cost value: {result.fun:.6f}")
    if hasattr(result, 'message'):
        print(f"Optimization message: {result.message}")

def save_optimization_report(result, error, time_array, theta_real, theta_sim, filename="reports/GS_optimization_report.txt"):
    """Save comprehensive optimization results to a file"""
    # Create twin to get g and l values
    twin = ModifiedDigitalTwin()
    
    # Unpack parameters
    I_scale, damping_coefficient, mass = result.x
    l = twin.l  # pendulum length from class
    g = twin.g  # gravity from class
    I = I_scale * mass * l**2  # moment of inertia
    
    # Calculate natural frequency
    omega_n = np.sqrt(g/l)  # natural frequency in rad/s
    f_n = omega_n/(2*np.pi)  # natural frequency in Hz
    
    # Calculate damping ratio
    zeta = damping_coefficient/(2*np.sqrt(I*g/l))  # damping ratio
    
    # Calculate quality factor
    Q = 1/(2*zeta) if zeta > 0 else float('inf')
    
    # Calculate time constant
    tau = 1/(zeta*omega_n) if zeta > 0 else float('inf')
    
    # Calculate cost function breakdown
    min_len = min(len(theta_sim), len(theta_real))
    dt = time_array[1] - time_array[0]
    theta_dot_sim = np.gradient(theta_sim[:min_len], dt)
    
    # Time domain error
    max_amplitude = np.max(np.abs(theta_real[:min_len]))
    time_weights = np.exp(np.linspace(0, 1, min_len))
    time_domain_error = np.mean(time_weights * ((theta_sim[:min_len] - theta_real[:min_len])/max_amplitude)**2)
    
    # Frequency error
    n_points = 8192
    real_fft = fft(theta_real[:min_len], n_points)
    sim_fft = fft(theta_sim[:min_len], n_points)
    real_mag = np.abs(real_fft[:n_points//2])
    sim_mag = np.abs(sim_fft[:n_points//2])
    real_mag_norm = real_mag / np.max(real_mag)
    sim_mag_norm = sim_mag / np.max(sim_mag)
    freq = np.fft.fftfreq(n_points, d=dt)[:n_points//2]
    freq_mask = (freq >= 0.9) & (freq <= 1.1)
    freq_error = np.mean((real_mag_norm[freq_mask] - sim_mag_norm[freq_mask])**2)
    
    # Amplitude error
    real_peaks = signal.find_peaks(np.abs(theta_real[:min_len]), distance=15)[0]
    sim_peaks = signal.find_peaks(np.abs(theta_sim[:min_len]), distance=15)[0]
    n_peaks = min(len(real_peaks), len(sim_peaks))
    real_amplitudes = np.abs(theta_real[real_peaks[:n_peaks]]) / max_amplitude
    sim_amplitudes = np.abs(theta_sim[sim_peaks[:n_peaks]]) / max_amplitude
    amplitude_error = np.mean((real_amplitudes - sim_amplitudes)**2)
    
    # Decay error
    real_decay = real_amplitudes[1:] / real_amplitudes[:-1]
    sim_decay = sim_amplitudes[1:] / sim_amplitudes[:-1]
    decay_error = np.mean((real_decay - sim_decay)**2)
    
    # Statistical Analysis
    error_mean = np.mean(error)
    error_std = np.std(error)
    error_skew = stats.skew(error)
    error_kurtosis = stats.kurtosis(error)
    
    # Physical Validation
    # Calculate measured natural frequency from zero crossings
    zero_crossings = np.where(np.diff(np.signbit(theta_real[:min_len])))[0]
    if len(zero_crossings) >= 2:
        measured_period = 2 * (time_array[zero_crossings[-1]] - time_array[zero_crossings[0]]) / (len(zero_crossings) - 1)
        measured_freq = 1/measured_period
        freq_error_percent = abs(measured_freq - f_n)/f_n * 100
    else:
        measured_freq = f_n
        freq_error_percent = 0
    
    # Energy conservation analysis
    E_real = 0.5 * l**2 * theta_dot_real[:min_len]**2 + g * l * (1 - np.cos(theta_real[:min_len]))
    E_sim = 0.5 * l**2 * theta_dot_sim**2 + g * l * (1 - np.cos(theta_sim[:min_len]))
    energy_error = np.mean((E_sim - E_real)/np.max(E_real))**2
    
    # Create report content
    report = []
    report.append("PENDULUM OPTIMIZATION REPORT (GRID SEARCH + LOCAL OPTIMIZATION)")
    report.append("=" * 50)
    report.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    report.append("OPTIMIZATION PARAMETERS")
    report.append("-" * 50)
    report.append("Grid Search Parameters:")
    report.append(f"I_scale range: {I_SCALE_RANGE[0]:.4f} to {I_SCALE_RANGE[-1]:.4f} ({len(I_SCALE_RANGE)} points)")
    report.append(f"Damping range: {DAMPING_RANGE[0]:.6f} to {DAMPING_RANGE[-1]:.6f} ({len(DAMPING_RANGE)} points)")
    report.append(f"Mass range: {MASS_RANGE[0]:.4f} to {MASS_RANGE[-1]:.4f} ({len(MASS_RANGE)} points)")
    report.append("")
    report.append("Local Optimization Parameters:")
    report.append(f"Method: {LOCAL_OPT_METHOD}")
    report.append(f"Maximum Iterations: {MAX_ITERATIONS}")
    report.append(f"Tolerance: {TOLERANCE}")
    report.append("")
    
    report.append("PHYSICAL PARAMETERS")
    report.append("-" * 50)
    report.append(f"Mass (mp): {mass:.4f} kg")
    report.append(f"Length (l): {l:.4f} m")
    report.append(f"Moment of Inertia Scale: {I_scale:.4f}")
    report.append(f"Effective I: {I:.6f} kg⋅m²")
    report.append(f"Damping Coefficient: {damping_coefficient:.8f}")
    report.append("")
    
    report.append("DYNAMIC CHARACTERISTICS")
    report.append("-" * 50)
    report.append(f"Natural Frequency: {f_n:.4f} Hz")
    report.append(f"Measured Frequency: {measured_freq:.4f} Hz")
    report.append(f"Frequency Error: {freq_error_percent:.2f}%")
    report.append(f"Damping Ratio: {zeta:.4f}")
    report.append(f"Quality Factor: {Q:.2f}")
    report.append(f"Time Constant: {tau:.4f} s")
    report.append("")
    
    report.append("PERFORMANCE METRICS")
    report.append("-" * 50)
    report.append(f"RMS Error: {np.sqrt(np.mean(error**2)):.4f} rad")
    report.append(f"Max Absolute Error: {np.max(np.abs(error)):.4f} rad")
    report.append(f"Mean Error: {error_mean:.4f} rad")
    report.append(f"Error Std Dev: {error_std:.4f} rad")
    report.append(f"Error Skewness: {error_skew:.4f}")
    report.append(f"Error Kurtosis: {error_kurtosis:.4f}")
    report.append("")
    
    report.append("COST FUNCTION BREAKDOWN")
    report.append("-" * 50)
    report.append(f"Time Domain Error: {50*time_domain_error:.4f}")
    report.append(f"Frequency Error: {600*freq_error:.4f}")
    report.append(f"Amplitude Error: {200*amplitude_error:.4f}")
    report.append(f"Decay Error: {200*decay_error:.4f}")
    report.append(f"Energy Error: {energy_error:.4f}")
    report.append(f"Total Cost: {result.fun:.4f}")
    report.append("")
    
    report.append("OPTIMIZATION SUMMARY")
    report.append("-" * 50)
    report.append(f"Success: {result.success}")
    report.append(f"Number of evaluations: {result.nfev}")
    report.append(f"Number of iterations: {result.nit}")
    report.append(f"Final cost value: {result.fun:.6f}")
    if hasattr(result, 'message'):
        report.append(f"Optimization message: {result.message}")
    report.append("")
    
    # Add sensitivity analysis
    report.append("SENSITIVITY ANALYSIS")
    report.append("-" * 50)
    
    # Test I_scale variations
    report.append("I_scale variations:")
    for i in range(-2, 3):
        if i == 0:
            continue
        factor = 1 + i*0.1
        test_I_scale = I_scale * factor
        test_params = [test_I_scale, damping_coefficient, mass]
        test_cost = parallel_cost_function(test_params, time_array, theta_real, theta_dot_real, theta0, theta_dot0)
        report.append(f"I_scale={test_I_scale:.4f} (change: {i*10:+d}%) → Cost={test_cost:.6f}")
    
    # Test mass variations
    report.append("\nMass variations:")
    for i in range(-2, 3):
        if i == 0:
            continue
        factor = 1 + i*0.1
        test_mass = mass * factor
        test_params = [I_scale, damping_coefficient, test_mass]
        test_cost = parallel_cost_function(test_params, time_array, theta_real, theta_dot_real, theta0, theta_dot0)
        report.append(f"mass={test_mass:.4f} (change: {i*10:+d}%) → Cost={test_cost:.6f}")
    
    # Test damping coefficient variations
    report.append("\nDamping coefficient variations:")
    for i in range(-2, 3):
        if i == 0:
            continue
        factor = 1 + i*0.1
        test_damping = damping_coefficient * factor
        test_params = [I_scale, test_damping, mass]
        test_cost = parallel_cost_function(test_params, time_array, theta_real, theta_dot_real, theta0, theta_dot0)
        report.append(f"damping={test_damping:.8f} (change: {i*10:+d}%) → Cost={test_cost:.6f}")
    
    # Write report to file
    with open(filename, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Optimization report saved to {filename}")

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