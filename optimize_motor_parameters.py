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

# Optimization parameters
POPULATION_SIZE = 40
MAX_ITERATIONS = 300
CONVERGENCE_TOLERANCE = 0.0001
MUTATION_RATE = (0.5, 1.0)
RECOMBINATION_RATE = 0.7

# Define the CSV filename for data loading
csv_filename = "move_a_17.7"

class ModifiedDigitalTwin(DigitalTwin):
    """Extended DigitalTwin with motor parameter optimization"""
    
    def __init__(self):
        super().__init__()
        # Physical parameters
        self.g = 9.81  # gravity (m/s²)
        self.l = 0.35  # length (m)
        self.mp = 0.35  # mass (kg)
        self.I_scale = 0.7110  # moment of inertia scale (from parent class)
        self.damping_coefficient = 0.0  # damping coefficient
        self.a_m = 1.0  # motor force transfer coefficient
        self.c_air = 0.001  # Air friction coefficient (from parent class)
        self.c_c = 0.00562279  # Coulomb friction coefficient (from parent class)
        self.R_pulley = 0.009  # Pulley radius (from parent class)
        
        # Derived parameters
        self.I = self.I_scale * self.mp * self.l**2  # moment of inertia

    def simulate_key_press(self, theta0, theta_dot0, time_array, key_duration):
        """Simulate a key press action with given duration"""
        # Create time array that includes settling time before key press
        settle_time = 2.0  # Let the pendulum settle for 2 seconds
        pre_time = np.arange(0, settle_time, self.delta_t)  # Settling time
        full_time = np.concatenate([pre_time, time_array + settle_time])
        
        # Initialize arrays for storing results
        theta = np.zeros_like(full_time)
        theta_dot = np.zeros_like(full_time)
        
        # Set initial conditions at t=0
        theta[0] = theta0  # Start from the same position as real data
        theta_dot[0] = theta_dot0  # Start with the same velocity as real data
        self.theta = theta0
        self.theta_dot = theta_dot0
        self.time = 0
        self.steps = 0
        
        # Let the pendulum settle first (no motor force)
        for i in range(1, len(pre_time)):
            # Step without motor force
            self.future_motor_accelerations = []
            self.future_motor_velocities = []
            self.future_motor_positions = []
            theta[i], theta_dot[i], _, _ = self.step()
        
        # Apply the key press after settling
        self.update_motor_accelerations_real('left', key_duration/1000.0)  # Convert ms to seconds
        
        # Continue simulation after key press
        for i in range(len(pre_time), len(full_time)):
            theta[i], theta_dot[i], _, _ = self.step()
        
        # Return only the part that corresponds to the comparison time
        start_idx = len(pre_time)
        return theta[start_idx:]

# Data loading function
def load_real_data(filename=None):
    """Load and preprocess real pendulum data from CSV file"""
    if filename is None:
        filename = f"datasets/filtered_datasets/{csv_filename}_kalman_output.csv"
    
    # Load data
    df = pd.read_csv(filename)
    print(f"\nTotal data points: {len(df)}")
    print(f"Time range: {df['time_sec'].min():.4f} to {df['time_sec'].max():.4f} seconds")
    
    # Find where x_pivot_m first changes from 0
    x_pivot = df['x_pivot_m'].values
    start_idx = np.where(np.abs(x_pivot) > 0.0001)[0][0]  # First non-zero position (with small tolerance)
    
    # Get the time when movement starts
    movement_start_time = df['time_sec'].values[start_idx]
    
    # Print the state at movement start
    print(f"\nAt movement start (t={movement_start_time:.4f}s):")
    print(f"x_pivot_m: {x_pivot[start_idx]:.4f}")
    print(f"theta: {df['theta_kalman'].values[start_idx]:.4f}")
    print(f"theta_dot: {df['theta_dot_kalman'].values[start_idx]:.4f}")
    
    # Trim all data to start from movement start
    df = df[df['time_sec'] >= movement_start_time].reset_index(drop=True)
    
    # Extract time and angle data
    time_array = df['time_sec'].values - movement_start_time  # Start time from 0
    theta_real = df['theta_kalman'].values
    theta_dot_real = df['theta_dot_kalman'].values
    
    # Get initial conditions at movement start
    theta0 = theta_real[0]  # First value after trimming
    theta_dot0 = theta_dot_real[0]  # First value after trimming
    
    print(f"\nTrimmed data points: {len(df)}")
    print(f"Trimmed time range: 0 to {time_array[-1]:.4f} seconds")
    print(f"Initial conditions: theta0 = {theta0:.4f}, theta_dot0 = {theta_dot0:.4f}")
    
    return time_array, theta_real, theta_dot_real, theta0, theta_dot0

def parallel_cost_function(params, time_array, theta_real, theta_dot_real, theta0, theta_dot0):
    """Cost function for optimization with motor parameters"""
    # Unpack parameters
    a_m, key_duration = params
    
    # Create twin with these parameters
    twin = ModifiedDigitalTwin()
    twin.a_m = a_m
    
    # Simulate with these parameters
    theta_sim = twin.simulate_key_press(theta0, theta_dot0, time_array, key_duration)
    
    # Check for invalid simulation
    if np.any(np.isnan(theta_sim)) or np.any(np.isinf(theta_sim)):
        return 1e6
    
    min_len = min(len(theta_sim), len(theta_real))
    dt = time_array[1] - time_array[0]
    
    # Calculate velocity for simulation
    theta_dot_sim = np.gradient(theta_sim[:min_len], dt)
    
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
    
    # Print debug info occasionally
    if np.random.random() < 0.01:  # Print 1% of the time
        print(f"\nDebug - a_m: {a_m:.4f}, duration: {key_duration:.1f}ms")
        print(f"Errors - Time: {time_domain_error:.4f}, Freq: {freq_error:.4f}, Amp: {amplitude_error:.4f}")
    
    # Weighted sum of errors with adjusted weights
    total_error = (
        100 * time_domain_error +     # Time domain weight
        300 * freq_error +            # Frequency weight
        200 * amplitude_error +       # Amplitude weight
        100 * decay_error            # Decay weight
    )
    
    return total_error

def optimize_motor_params():
    """Run the optimization process using differential evolution"""
    # Load the real data
    time_array, theta_real, theta_dot_real, theta0, theta_dot0 = load_real_data()
    
    # Define parameter bounds
    bounds = [
        (0.1, 1.0),     # a_m: motor force coefficient
        (300, 600)      # key_duration (ms)
    ]
    
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
        maxiter=MAX_ITERATIONS,
        tol=CONVERGENCE_TOLERANCE,
        mutation=MUTATION_RATE,
        recombination=RECOMBINATION_RATE,
        seed=42,
        workers=n_cores,
        updating='deferred'
    )
    
    return result

def simulate_and_plot(params):
    """Simulate and plot results with given parameters"""
    # Load real data
    time_array, theta_real, theta_dot_real, theta0, theta_dot0 = load_real_data()
    
    # Create twin with optimized parameters
    twin = ModifiedDigitalTwin()
    a_m, key_duration = params
    twin.a_m = a_m
    
    # Simulate
    theta_sim = twin.simulate_key_press(theta0, theta_dot0, time_array, key_duration)
    
    # Calculate error and derivatives
    min_len = min(len(theta_sim), len(theta_real))
    dt = time_array[1] - time_array[0]
    theta_dot_sim = np.gradient(theta_sim[:min_len], dt)
    error = theta_sim[:min_len] - theta_real[:min_len]
    
    # Create comprehensive analysis plots
    plt.figure(figsize=(20, 20))
    
    # 1. Time-domain position comparison
    plt.subplot(4, 2, 1)
    plt.plot(time_array[:min_len], theta_real[:min_len], 'b-', label='Real θ')
    plt.plot(time_array[:min_len], theta_sim[:min_len], 'r--', label='Simulated θ')
    plt.axvline(x=key_duration/1000.0, color='g', linestyle='--', label=f'Key Press End ({key_duration}ms)')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.title('Pendulum Angle: Real vs Simulated')
    plt.grid(True)
    plt.legend()
    
    # 2. Velocity comparison
    plt.subplot(4, 2, 2)
    plt.plot(time_array[:min_len], theta_dot_real[:min_len], 'b-', label='Real θ̇')
    plt.plot(time_array[:min_len], theta_dot_sim[:min_len], 'r--', label='Simulated θ̇')
    plt.axvline(x=key_duration/1000.0, color='g', linestyle='--', label=f'Key Press End ({key_duration}ms)')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.title('Angular Velocity Comparison')
    plt.grid(True)
    plt.legend()
    
    # 3. Phase space plot
    plt.subplot(4, 2, 3)
    plt.plot(theta_real[:min_len], theta_dot_real[:min_len], 'b-', label='Real', alpha=0.5)
    plt.plot(theta_sim[:min_len], theta_dot_sim[:min_len], 'r--', label='Simulated', alpha=0.5)
    plt.xlabel('θ (rad)')
    plt.ylabel('θ̇ (rad/s)')
    plt.title('Phase Space Plot')
    plt.grid(True)
    plt.legend()
    
    # 4. Error analysis
    plt.subplot(4, 2, 4)
    plt.plot(time_array[:min_len], error, 'g-', label='Error')
    plt.axvline(x=key_duration/1000.0, color='g', linestyle='--', label=f'Key Press End ({key_duration}ms)')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (rad)')
    plt.title('Simulation Error')
    plt.grid(True)
    plt.legend()
    
    # 5. Frequency analysis
    n_points = 8192
    real_fft = fft(theta_real[:min_len], n_points)
    sim_fft = fft(theta_sim[:min_len], n_points)
    freq = np.fft.fftfreq(n_points, d=dt)[:n_points//2]
    
    plt.subplot(4, 2, 5)
    plt.plot(freq, np.abs(real_fft[:n_points//2]), 'b-', label='Real', alpha=0.5)
    plt.plot(freq, np.abs(sim_fft[:n_points//2]), 'r--', label='Simulated', alpha=0.5)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Frequency Spectrum')
    plt.grid(True)
    plt.legend()
    
    # 6. Energy analysis
    l = twin.l
    g = twin.g
    E_real = 0.5 * l**2 * theta_dot_real[:min_len]**2 + g * l * (1 - np.cos(theta_real[:min_len]))
    E_sim = 0.5 * l**2 * theta_dot_sim**2 + g * l * (1 - np.cos(theta_sim[:min_len]))
    
    plt.subplot(4, 2, 6)
    plt.plot(time_array[:min_len], E_real, 'b-', label='Real', alpha=0.5)
    plt.plot(time_array[:min_len], E_sim, 'r--', label='Simulated', alpha=0.5)
    plt.axvline(x=key_duration/1000.0, color='g', linestyle='--', label=f'Key Press End ({key_duration}ms)')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (J)')
    plt.title('System Energy')
    plt.grid(True)
    plt.legend()
    
    # 7. Motor force profile
    plt.subplot(4, 2, 7)
    motor_force = np.zeros_like(time_array[:min_len])
    motor_force[time_array[:min_len] <= key_duration/1000.0] = -a_m
    plt.plot(time_array[:min_len], motor_force, 'b-', label='Motor Force')
    plt.axvline(x=key_duration/1000.0, color='g', linestyle='--', label=f'Key Press End ({key_duration}ms)')
    plt.xlabel('Time (s)')
    plt.ylabel('Motor Force')
    plt.title('Motor Force Profile')
    plt.grid(True)
    plt.legend()
    
    # 8. Error distribution
    plt.subplot(4, 2, 8)
    plt.hist(error, bins=50, density=True, alpha=0.7, color='g')
    plt.xlabel('Error (rad)')
    plt.ylabel('Density')
    plt.title('Error Distribution')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('reports/optimization_results.png', dpi=300, bbox_inches='tight')
    plt.show()  # This will keep the plot window open
    
    return theta_sim, error, a_m, key_duration

def print_optimization_results(result, error, time_array, theta_real, theta_sim):
    """Print comprehensive optimization results"""
    # Unpack parameters
    a_m, key_duration = result.x
    
    # Calculate error metrics
    min_len = min(len(theta_sim), len(theta_real))
    dt = time_array[1] - time_array[0]
    theta_dot_sim = np.gradient(theta_sim[:min_len], dt)
    
    # Time domain error
    max_amplitude = np.max(np.abs(theta_real[:min_len]))
    time_weights = np.exp(np.linspace(0, 1, min_len))
    time_domain_error = np.mean(time_weights * ((theta_sim[:min_len] - theta_real[:min_len])/max_amplitude)**2)
    
    # Print results
    print("\nOptimization Results:")
    print("-" * 50)
    print("Motor Parameters:")
    print(f"Motor Force Transfer Coefficient (a_m): {a_m:.4f}")
    print(f"Key Press Duration: {key_duration:.1f} ms")
    print(f"Final Error: {result.fun:.6f}")
    print(f"Time Domain Error: {time_domain_error:.6f}")

def save_optimization_report(result, error, time_array, theta_real, theta_sim, filename="reports/motor_optimization_report.txt"):
    """Save comprehensive optimization results to a file"""
    # Unpack parameters
    a_m, key_duration = result.x
    
    # Calculate error metrics
    min_len = min(len(theta_sim), len(theta_real))
    dt = time_array[1] - time_array[0]
    theta_dot_sim = np.gradient(theta_sim[:min_len], dt)
    
    # Time domain error
    max_amplitude = np.max(np.abs(theta_real[:min_len]))
    time_weights = np.exp(np.linspace(0, 1, min_len))
    time_domain_error = np.mean(time_weights * ((theta_sim[:min_len] - theta_real[:min_len])/max_amplitude)**2)
    
    # Create report content
    report = []
    report.append("MOTOR PARAMETER OPTIMIZATION REPORT")
    report.append("=" * 50)
    report.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    report.append("OPTIMIZATION PARAMETERS")
    report.append("-" * 50)
    report.append(f"Population Size: {POPULATION_SIZE}")
    report.append(f"Maximum Iterations: {MAX_ITERATIONS}")
    report.append(f"Convergence Tolerance: {CONVERGENCE_TOLERANCE}")
    report.append(f"Mutation Rate: {MUTATION_RATE}")
    report.append(f"Recombination Rate: {RECOMBINATION_RATE}")
    report.append("")
    
    report.append("OPTIMIZATION RESULTS")
    report.append("-" * 50)
    report.append(f"Motor Force Transfer Coefficient (a_m): {a_m:.4f}")
    report.append(f"Key Press Duration: {key_duration:.1f} ms")
    report.append(f"Final Error: {result.fun:.6f}")
    report.append(f"Time Domain Error: {time_domain_error:.6f}")
    
    # Write report to file
    with open(filename, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Optimization report saved to {filename}")

# Main execution
if __name__ == "__main__":
    # Load the real data first - this will be used throughout
    time_array, theta_real, theta_dot_real, theta0, theta_dot0 = load_real_data()
    
    # Run the optimization
    result = optimize_motor_params()
    
    # Get the best parameters
    a_m, key_duration = result.x
    
    # Simulate and plot with best parameters
    theta_sim, error, a_m, key_duration = simulate_and_plot(result.x)
    
    # Print comprehensive results
    print_optimization_results(result, error, time_array, theta_real, theta_sim)
    
    # Save optimization report
    save_optimization_report(result, error, time_array, theta_real, theta_sim) 