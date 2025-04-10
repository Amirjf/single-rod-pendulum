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
    
    # Use velocity directly from the CSV file (from kalman_filter_plots_mina)
    theta_dot_real = df_trimmed['theta_dot_kalman'].values
    
    # Get initial conditions
    theta0 = theta_real[0]
    theta_dot0 = theta_dot_real[0]
    
    print(f"Initial conditions: theta0 = {theta0:.4f}, theta_dot0 = {theta_dot0:.4f}")
    return time_array, theta_real, theta_dot_real, theta0, theta_dot0

def parallel_cost_function(params, time_array, theta_real, theta_dot_real, theta0, theta_dot0):
    """Cost function for optimization with mass-independent error metrics"""
    # Unpack parameters
    I_scale, damping_coefficient, mass = params
    
    # Create twin with these parameters
    twin = ModifiedDigitalTwin()
    twin.mp = mass
    twin.l = 0.35
    twin.I_scale = I_scale
    twin.I = I_scale * twin.mp * twin.l**2
    twin.damping_coefficient = damping_coefficient
    twin.c_air = 0.0
    twin.currentmotor_acceleration = 0
    
    # Simulate with these parameters
    theta_sim = twin.simulate_passive(theta0, theta_dot0, time_array)
    
    # Check for invalid simulation
    if np.any(np.isnan(theta_sim)) or np.any(np.isinf(theta_sim)):
        return 1e6
    
    min_len = min(len(theta_sim), len(theta_real))
    dt = time_array[1] - time_array[0]
    
    # Calculate theta_dot for simulation using the same method as kalman_filter_plots_mina.ipynb
    # 1. Finite difference
    theta_dot_sim_raw = np.gradient(theta_sim[:min_len], dt)
    # 2. Gaussian smoothing (similar to kalman_filter_plots_mina.ipynb)
    theta_dot_sim = gaussian_filter1d(theta_dot_sim_raw, sigma=2)
    
    # 1. Time-domain position error with exponential weighting
    time_weights = np.exp(np.linspace(0, 1, min_len))
    max_amplitude = np.max(np.abs(theta_real[:min_len]))
    time_domain_error = np.mean(time_weights * ((theta_sim[:min_len] - theta_real[:min_len])/max_amplitude)**2)
    
    # 2. Frequency analysis
    n_points = 8192
    real_fft = fft(theta_real[:min_len], n_points)
    sim_fft = fft(theta_sim[:min_len], n_points)
    
    real_mag = np.abs(real_fft[:n_points//2])
    sim_mag = np.abs(sim_fft[:n_points//2])
    
    real_mag_norm = real_mag / np.max(real_mag)
    sim_mag_norm = sim_mag / np.max(sim_mag)
    
    freq = np.fft.fftfreq(n_points, d=dt)[:n_points//2]
    freq_mask = (freq >= 0.9) & (freq <= 1.1)
    
    # Frequency matching error
    freq_error = np.mean((real_mag_norm[freq_mask] - sim_mag_norm[freq_mask])**2)
    
    # 3. Peak amplitude analysis
    real_peaks = signal.find_peaks(np.abs(theta_real[:min_len]), distance=15)[0]
    sim_peaks = signal.find_peaks(np.abs(theta_sim[:min_len]), distance=15)[0]
    
    if len(real_peaks) < 3 or len(sim_peaks) < 3:
        return 1e6
    
    n_peaks = min(len(real_peaks), len(sim_peaks))
    real_amplitudes = np.abs(theta_real[real_peaks[:n_peaks]]) / max_amplitude
    sim_amplitudes = np.abs(theta_sim[sim_peaks[:n_peaks]]) / max_amplitude
    
    # Amplitude error
    amplitude_error = np.mean((real_amplitudes - sim_amplitudes)**2)
    
    # Calculate amplitude decay rates
    real_decay = real_amplitudes[1:] / real_amplitudes[:-1]
    sim_decay = sim_amplitudes[1:] / sim_amplitudes[:-1]
    decay_error = np.mean((real_decay - sim_decay)**2)
    
    # Weighted sum of errors
    total_error = (
        50 * time_domain_error +      # Time domain weight
        600 * freq_error +            # Frequency weight
        200 * amplitude_error +       # Amplitude weight
        200 * decay_error             # Decay weight
    )
    
    return total_error

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
    
    # Print best results
    print("\nGrid Search Results:")
    print("-" * 50)
    print(f"Best parameters found:")
    print(f"I_scale: {best_params[0]:.6f}")
    print(f"Damping: {best_params[1]:.6f}")
    print(f"Mass: {best_params[2]:.6f}")
    print(f"Best cost: {best_cost:.6f}")
    
    # Create a result object for consistency with other optimization methods
    class OptimizeResult:
        pass
    
    result = OptimizeResult()
    result.x = np.array(best_params)
    result.fun = best_cost
    result.success = True
    result.nfev = total_points
    result.nit = 1
    result.message = "Grid search optimization completed successfully."
    
    return result

def plot_comprehensive_analysis(theta_real, theta_sim, time_array, theta_dot_real=None):
    """Create comprehensive analysis plots with multiple metrics"""
    min_len = min(len(theta_sim), len(theta_real))
    dt = time_array[1] - time_array[0]
    
    # Create twin to get g and l values
    twin = ModifiedDigitalTwin()
    
    # Calculate theta_dot for simulation using the same method as kalman_filter_plots_mina.ipynb
    # 1. Finite difference
    theta_dot_sim_raw = np.gradient(theta_sim, dt)
    # 2. Gaussian smoothing (similar to kalman_filter_plots_mina.ipynb)
    theta_dot_sim = gaussian_filter1d(theta_dot_sim_raw, sigma=2)
    
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
    
    # 2. Velocity comparison - use theta_dot_real directly from CSV
    ax2 = fig.add_subplot(4, 2, 2)
    ax2.plot(time_array[:min_len], theta_dot_real[:min_len], 'b-', label='Real θ̇ (Kalman)')
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

def simulate_and_plot(params, time_array=None, theta_real=None, theta_dot_real=None, theta0=None, theta_dot0=None):
    """Simulate pendulum motion and create analysis plots"""
    if time_array is None:
        time_array, theta_real, theta_dot_real, theta0, theta_dot0 = load_real_data()
    
    # Unpack parameters
    I_scale, damping_coefficient, mass = params
    
    # Create twin with these parameters
    twin = ModifiedDigitalTwin()
    twin.mp = mass
    twin.l = 0.35
    twin.I_scale = I_scale
    twin.I = I_scale * twin.mp * twin.l**2
    twin.damping_coefficient = damping_coefficient
    twin.c_air = 0.0
    twin.currentmotor_acceleration = 0
    
    # Simulate with these parameters
    theta_sim = twin.simulate_passive(theta0, theta_dot0, time_array)
    
    # Calculate theta_dot for simulation using the same method as kalman_filter_plots_mina.ipynb
    dt = time_array[1] - time_array[0]
    theta_dot_sim_raw = np.gradient(theta_sim, dt)
    theta_dot_sim = gaussian_filter1d(theta_dot_sim_raw, sigma=2)
    
    # Create comprehensive analysis plots
    fig = plot_comprehensive_analysis(theta_real, theta_sim, time_array, theta_dot_real)
    
    # Calculate metrics
    min_len = min(len(theta_sim), len(theta_real))
    error = theta_sim[:min_len] - theta_real[:min_len]
    rms_error = np.sqrt(np.mean(error**2))
    max_error = np.max(np.abs(error))
    
    # Add parameter box
    add_parameter_box(fig, params, twin, rms_error, max_error)
    
    # Save plot in reports folder with GS-specific naming
    plt.savefig('reports/half_theta_2_pendulum_GS_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return theta_sim, error

def add_parameter_box(fig, params, twin=None, rms_error=None, max_error=None):
    """Add a text box with model parameters to the figure"""
    # Unpack parameters
    I_scale, damping_coefficient, mass = params
    
    # Create twin if not provided
    if twin is None:
        twin = ModifiedDigitalTwin()
        twin.mp = mass
        twin.l = 0.35
        twin.I_scale = I_scale
        twin.I = I_scale * twin.mp * twin.l**2
        twin.damping_coefficient = damping_coefficient
    
    # Calculate dynamic characteristics
    g = 9.81  # gravity
    L = twin.l  # pendulum length
    I = twin.I  # moment of inertia
    
    # Natural frequency
    omega_n = np.sqrt(g/L)  # rad/s
    f_n = omega_n/(2*np.pi)  # Hz
    
    # Damping ratio
    zeta = damping_coefficient/(2*np.sqrt(I*g/L))
    
    # Quality factor
    Q = 1/(2*zeta) if zeta != 0 else float('inf')
    
    # Create parameter text
    param_text = (
        f"Parameters:\n"
        f"I_scale: {I_scale:.4f}\n"
        f"Damping: {damping_coefficient:.6f}\n"
        f"Mass: {mass:.4f} kg\n"
        f"I: {I:.6f} kg⋅m²\n"
        f"f_n: {f_n:.4f} Hz\n"
        f"ζ: {zeta:.4f}\n"
        f"Q: {Q:.1f}"
    )
    
    if rms_error is not None:
        param_text += f"\nRMS Error: {rms_error:.6f}"
    if max_error is not None:
        param_text += f"\nMax Error: {max_error:.6f}"
    
    # Add text box
    plt.figtext(0.02, 0.02, param_text, fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8))

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
    
    print("\nSensitivity Metrics (max increase/decrease relative to base):")
    print(f"I_scale:     +{I_sensitivity[0]*100:.1f}% / -{I_sensitivity[1]*100:.1f}%")
    print(f"Mass:        +{mass_sensitivity[0]*100:.1f}% / -{mass_sensitivity[1]*100:.1f}%")
    print(f"Damping:     +{damping_sensitivity[0]*100:.1f}% / -{damping_sensitivity[1]*100:.1f}%")

def print_optimization_results(result, error, time_array, theta_real, theta_sim):
    """Print comprehensive optimization results."""
    # Unpack parameters
    I_scale, damping_coefficient, mass = result.x
    
    # Calculate dynamic characteristics
    L = 0.35  # pendulum length
    g = 9.81  # gravity
    I = I_scale * mass * L**2  # moment of inertia
    
    # Natural frequency
    omega_n = np.sqrt(g/L)  # rad/s
    f_n = omega_n/(2*np.pi)  # Hz
    
    # Damping ratio
    zeta = damping_coefficient/(2*np.sqrt(I*g/L))
    
    # Quality factor
    Q = 1/(2*zeta) if zeta != 0 else float('inf')
    
    # Calculate errors
    rms_error = np.sqrt(np.mean(error**2))
    max_error = np.max(np.abs(error))
    
    # Print results
    print("\nOptimization Results:")
    print("-" * 50)
    print("Physical Parameters:")
    print(f"Mass (m): {mass:.4f} kg")
    print(f"Length (L): {L:.4f} m")
    print(f"Moment of Inertia Scale: {I_scale:.4f}")
    print(f"Effective I: {I:.6f} kg⋅m²")
    print(f"Damping Coefficient: {damping_coefficient:.6f} N⋅m⋅s/rad")
    
    print("\nDynamic Characteristics:")
    print(f"Natural Frequency: {f_n:.4f} Hz")
    print(f"Damping Ratio: {zeta:.4f}")
    print(f"Quality Factor: {Q:.4f}")
    
    print("\nPerformance Metrics:")
    print(f"RMS Error: {rms_error:.6f} rad")
    print(f"Max Error: {max_error:.6f} rad")
    print(f"Final Cost: {result.fun:.6f}")
    
    print("\nOptimization Info:")
    print(f"Success: {result.success}")
    print(f"Number of evaluations: {result.nfev}")
    print(f"Number of iterations: {result.nit}")
    print(f"Final cost value: {result.fun:.6f}")
    if hasattr(result, 'message'):
        print(f"Optimization message: {result.message}")

def save_optimization_report(result, error, time_array, theta_real, theta_sim):
    """Save optimization results to a text file."""
    # Prepare report content
    report = []
    report.append("GRID SEARCH OPTIMIZATION REPORT")
    report.append("=" * 50)
    
    # Unpack parameters
    I_scale, damping_coefficient, mass = result.x
    
    # Calculate dynamic characteristics
    L = 0.35  # pendulum length
    g = 9.81  # gravity
    I = I_scale * mass * L**2  # moment of inertia
    
    # Natural frequency
    omega_n = np.sqrt(g/L)  # rad/s
    f_n = omega_n/(2*np.pi)  # Hz
    
    # Damping ratio
    zeta = damping_coefficient/(2*np.sqrt(I*g/L))
    
    # Quality factor
    Q = 1/(2*zeta) if zeta != 0 else float('inf')
    
    # Calculate errors
    rms_error = np.sqrt(np.mean(error**2))
    max_error = np.max(np.abs(error))
    
    # Add results to report
    report.append("\nPhysical Parameters:")
    report.append(f"Mass (m): {mass:.4f} kg")
    report.append(f"Length (L): {L:.4f} m")
    report.append(f"Moment of Inertia Scale: {I_scale:.4f}")
    report.append(f"Effective I: {I:.6f} kg⋅m²")
    report.append(f"Damping Coefficient: {damping_coefficient:.6f} N⋅m⋅s/rad")
    
    report.append("\nDynamic Characteristics:")
    report.append(f"Natural Frequency: {f_n:.4f} Hz")
    report.append(f"Damping Ratio: {zeta:.4f}")
    report.append(f"Quality Factor: {Q:.4f}")
    
    report.append("\nPerformance Metrics:")
    report.append(f"RMS Error: {rms_error:.6f} rad")
    report.append(f"Max Error: {max_error:.6f} rad")
    report.append(f"Final Cost: {result.fun:.6f}")
    
    report.append("\nOptimization Info:")
    report.append(f"Success: {result.success}")
    report.append(f"Number of evaluations: {result.nfev}")
    report.append(f"Number of iterations: {result.nit}")
    report.append(f"Final cost value: {result.fun:.6f}")
    if hasattr(result, 'message'):
        report.append(f"\nOptimization message: {result.message}")
    
    # Save report
    with open('reports/GS_optimization_report.txt', 'w') as f:
        f.write('\n'.join(report))

# Main execution
if __name__ == "__main__":
    # Load the real data first - this will be used throughout
    time_array, theta_real, theta_dot_real, theta0, theta_dot0 = load_real_data()
    
    # Run the grid search optimization
    result = grid_search_optimize()
    
    # Get the best parameters
    I_scale, damping_coefficient, mass = result.x
    
    # Create twin with these parameters
    twin = ModifiedDigitalTwin()
    twin.mp = mass
    twin.l = 0.35
    twin.I_scale = I_scale
    twin.I = I_scale * twin.mp * twin.l**2
    twin.damping_coefficient = damping_coefficient
    twin.c_air = 0.0
    twin.currentmotor_acceleration = 0
    
    # Simulate with these parameters
    theta_sim = twin.simulate_passive(theta0, theta_dot0, time_array)
    
    # Calculate error
    min_len = min(len(theta_sim), len(theta_real))
    error = theta_sim[:min_len] - theta_real[:min_len]
    
    # Simulate and plot with best parameters
    theta_sim, error = simulate_and_plot(result.x, time_array, theta_real, theta_dot_real, theta0, theta_dot0)
    
    # Print comprehensive results
    print_optimization_results(result, error, time_array, theta_real, theta_sim)
    
    # Save optimization report
    save_optimization_report(result, error, time_array, theta_real, theta_sim)
    
    # Analyze parameter sensitivity
    analyze_parameter_sensitivity(result, time_array, theta_real, theta_dot_real, theta0, theta_dot0)
