import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy import signal
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from Digital_twin import DigitalTwin
import time
import os

# Define the CSV filename for data loading
csv_filename = "half_theta_2"

# Helper functions for common calculations
def create_twin_with_params(mass, I_scale, damping_coefficient, use_pygame=True):
    """Create and initialize a DigitalTwin with given parameters to match ModifiedDigitalTwin behavior"""
    twin = DigitalTwin()
    twin.mp = mass
    twin.l = 0.35
    twin.I_scale = I_scale
    twin.I = I_scale * twin.mp * twin.l**2
    # Set damping coefficient to match ModifiedDigitalTwin behavior
    twin.c_c = damping_coefficient  # Use c_c instead of damping_coefficient
    twin.c_air = 0.0
    twin.currentmotor_acceleration = 0
    
    # Set Pygame visualization flag
    twin.use_pygame = use_pygame
    
    return twin

def calculate_fft_analysis(theta, dt, n_points=8192):
    """Calculate FFT analysis for given theta data"""
    fft_result = fft(theta, n_points)
    freq = np.fft.fftfreq(n_points, d=dt)[:n_points//2]
    mag = np.abs(fft_result[:n_points//2])
    mag_norm = mag / np.max(mag)
    return mag_norm, freq

def calculate_velocity(theta, dt):
    """Calculate velocity using gradient and gaussian smoothing"""
    theta_dot_raw = np.gradient(theta, dt)
    return gaussian_filter1d(theta_dot_raw, sigma=2)

def calculate_errors(theta_sim, theta_real, theta_dot_sim=None, theta_dot_real=None):
    """Calculate various error metrics between simulated and real data"""
    min_len = min(len(theta_sim), len(theta_real))
    position_error = theta_sim[:min_len] - theta_real[:min_len]
    rms_error = np.sqrt(np.mean(position_error**2))
    max_error = np.max(np.abs(position_error))
    
    if theta_dot_sim is not None and theta_dot_real is not None:
        velocity_error = theta_dot_sim[:min_len] - theta_dot_real[:min_len]
        return position_error, velocity_error, rms_error, max_error
    return position_error, rms_error, max_error

def find_peaks_and_zeros(theta, time_array, min_len):
    """Find peaks and zero crossings in theta data"""
    peaks, _ = signal.find_peaks(np.abs(theta[:min_len]), distance=15)
    zeros = np.where(np.diff(np.signbit(theta[:min_len])))[0]
    return peaks, zeros

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
    twin = create_twin_with_params(mass, I_scale, damping_coefficient)
    
    # Simulate with these parameters
    theta_sim = twin.simulate_passive(theta0, theta_dot0, time_array)
    
    # Check for invalid simulation
    if np.any(np.isnan(theta_sim)) or np.any(np.isinf(theta_sim)):
        return 1e6
    
    min_len = min(len(theta_sim), len(theta_real))
    dt = time_array[1] - time_array[0]
    
    # Calculate theta_dot for simulation
    theta_dot_sim = calculate_velocity(theta_sim[:min_len], dt)
    
    # 1. Time-domain position error with exponential weighting
    time_weights = np.exp(np.linspace(0, 1, min_len))
    max_amplitude = np.max(np.abs(theta_real[:min_len]))
    time_domain_error = np.mean(time_weights * ((theta_sim[:min_len] - theta_real[:min_len])/max_amplitude)**2)
    
    # 2. Frequency analysis
    real_mag_norm, freq = calculate_fft_analysis(theta_real[:min_len], dt)
    sim_mag_norm, _ = calculate_fft_analysis(theta_sim[:min_len], dt)
    
    freq_mask = (freq >= 0.9) & (freq <= 1.1)
    freq_error = np.mean((real_mag_norm[freq_mask] - sim_mag_norm[freq_mask])**2)
    
    # 3. Peak amplitude analysis
    real_peaks, _ = find_peaks_and_zeros(theta_real, time_array, min_len)
    sim_peaks, _ = find_peaks_and_zeros(theta_sim, time_array, min_len)
    
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

def analyze_parameter_sensitivity(result, time_array, theta_real, theta_dot_real, theta0, theta_dot0):
    """
    Analyze the sensitivity of the cost function to parameter variations.
    
    Args:
        result: Optimization result object
        time_array: Array of time points
        theta_real: Real theta data
        theta_dot_real: Real theta_dot data
        theta0: Initial theta
        theta_dot0: Initial theta_dot
        
    Returns:
        Dictionary containing sensitivity analysis results
    """
    # Get best parameters from result
    if hasattr(result, 'x'):
        best_params = result.x
    else:
        best_params = result
        
    # Create a copy of best parameters for testing
    test_params = best_params.copy() if hasattr(best_params, 'copy') else best_params
    
    # Calculate base cost
    base_cost = parallel_cost_function(test_params, time_array, theta_real, theta_dot_real, theta0, theta_dot0)
    
    # Parameter variations to test (10% change)
    variation = 0.1
    
    sensitivities = {}
    param_names = ['I_scale', 'damping', 'mass']
    
    for i, param_name in enumerate(param_names):
        # Test positive variation
        test_params[i] = best_params[i] * (1 + variation)
        pos_cost = parallel_cost_function(test_params, time_array, theta_real, theta_dot_real, theta0, theta_dot0)
        
        # Test negative variation
        test_params[i] = best_params[i] * (1 - variation)
        neg_cost = parallel_cost_function(test_params, time_array, theta_real, theta_dot_real, theta0, theta_dot0)
        
        # Calculate sensitivity as average cost change
        sensitivity = (abs(pos_cost - base_cost) + abs(neg_cost - base_cost)) / (2 * base_cost)
        sensitivities[param_name] = sensitivity
        
        # Reset parameter
        test_params[i] = best_params[i]
    
    return sensitivities

def plot_comprehensive_analysis(theta_real, theta_sim, time_array, theta_dot_real=None, title="", model_name="", params=None):
    """Create comprehensive analysis plots with multiple metrics"""
    min_len = min(len(theta_sim), len(theta_real))
    dt = time_array[1] - time_array[0]
    
    # Calculate theta_dot for simulation
    theta_dot_sim = calculate_velocity(theta_sim, dt)
    
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
    ax2.plot(time_array[:min_len], theta_dot_real[:min_len], 'b-', label='Real θ̇ (from CSV)')
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
    real_peaks, real_zeros = find_peaks_and_zeros(theta_real, time_array, min_len)
    sim_peaks, sim_zeros = find_peaks_and_zeros(theta_sim, time_array, min_len)
    
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
    position_error, velocity_error, _, _ = calculate_errors(theta_sim, theta_real, theta_dot_sim, theta_dot_real)
    
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
    real_mag_norm, freq = calculate_fft_analysis(theta_real[:min_len], dt)
    sim_mag_norm, _ = calculate_fft_analysis(theta_sim[:min_len], dt)
    
    ax7.plot(freq, real_mag_norm, 'b-', label='Real Data')
    ax7.plot(freq, sim_mag_norm, 'r--', label='Simulation')
    ax7.set_xlabel('Frequency (Hz)')
    ax7.set_ylabel('Magnitude')
    ax7.set_title('Frequency Domain Analysis')
    ax7.set_xlim(0, 2)
    ax7.grid(True)
    ax7.legend()
    
    # 8. Theta Error over Time
    ax8 = fig.add_subplot(4, 2, 8)
    theta_error = theta_sim[:min_len] - theta_real[:min_len]
    ax8.plot(time_array[:min_len], theta_error, 'g-', label='θ Error')
    ax8.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Error (rad)')
    ax8.set_title('Theta Error Over Time')
    ax8.grid(True)
    ax8.legend()
    
    plt.tight_layout()
    
    # Calculate error metrics
    position_error, rms_error, max_error = calculate_errors(theta_sim, theta_real)
    
    if params is not None:
        # Create twin with parameters
        twin = create_twin_with_params(params[2], params[0], params[1])
        
        # Add parameter box
        add_parameter_box(fig, params, twin, rms_error, max_error)
    
    # Save the figure
    if model_name:
        plt.savefig(f'reports/{title}_pendulum_{model_name}_analysis.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f'reports/{title}_pendulum_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def simulate_and_plot(params, time_array, theta_real, theta_dot_real, theta0, theta_dot0, title, model_name):
    """Simulate pendulum with given parameters and plot results"""
    # Create twin with these parameters
    I_scale, damping_coefficient, mass = params
    twin = create_twin_with_params(mass, I_scale, damping_coefficient)
    
    # Simulate with these parameters
    theta_sim = twin.simulate_passive(theta0, theta_dot0, time_array)
    
    # Calculate errors
    position_error, rms_error, max_error = calculate_errors(theta_sim, theta_real)
    
    # Create comprehensive analysis plots
    fig = plot_comprehensive_analysis(theta_real, theta_sim, time_array, theta_dot_real, title, model_name=model_name, params=params)
    
    return theta_sim, position_error, params

def add_parameter_box(fig, params, twin=None, rms_error=None, max_error=None):
    """Add a comprehensive parameter information box to the figure.
    
    Args:
        fig: matplotlib figure object
        params: list of [I_scale, damping_coefficient, mass]
        twin: DigitalTwin object (optional)
        rms_error: RMS error value (optional)
        max_error: Maximum error value (optional)
    """
    # Unpack parameters
    I_scale, damping_coefficient, mass = params
    
    # Create twin if not provided
    if twin is None:
        twin = create_twin_with_params(mass, I_scale, damping_coefficient)
    
    # Calculate dynamic characteristics
    g = 9.81  # gravity
    L = twin.l  # pendulum length
    I = twin.I  # moment of inertia
    
    # Natural frequency
    omega_n = np.sqrt(g/L)  # rad/s
    f_n = omega_n/(2*np.pi)  # Hz
    
    # Damping ratio
    zeta = twin.c_c/(2*np.sqrt(I*g/L))
    
    # Quality factor
    Q = 1/(2*zeta) if zeta != 0 else float('inf')
    
    # Create parameter text
    param_text = (
        f'Model Parameters:\n'
        f'I_scale = {I_scale:.6f}\n'
        f'damping_coefficient = {twin.c_c:.6f}\n'
        f'mass = {mass:.6f} kg\n'
        f'\nPhysical Values:\n'
        f'Length = {L:.3f} m\n'
        f'I_eff = {I:.6f} kg⋅m²\n'
        f'\nDynamic Characteristics:\n'
        f'Natural Frequency = {f_n:.4f} Hz\n'
        f'Damping Ratio = {zeta:.4f}\n'
        f'Quality Factor = {Q:.1f}'
    )
    
    # Add performance metrics if provided
    if rms_error is not None:
        param_text += f'\n\nPerformance:\nRMS Error = {rms_error:.6f} rad'
    if max_error is not None:
        param_text += f'\nMax Error = {max_error:.6f} rad'
    
    # Add text box
    fig.text(0.98, 0.98, param_text, fontsize=10, family='monospace',
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.8)) 

def generate_optimization_report(best_params, time_array, theta_real, theta_dot_real, theta0, theta_dot0, 
                                title="", model_name="", cost_value=None, n_evaluations=None, 
                                optimization_time=None, sensitivities=None):
    """
    Generate a comprehensive report of optimization results and save it as a text file.
    
    Args:
        best_params: List of [I_scale, damping_coefficient, mass]
        time_array: Time array from real data
        theta_real: Real theta data
        theta_dot_real: Real theta_dot data
        theta0: Initial theta
        theta_dot0: Initial theta_dot
        title: Title for the report
        model_name: Name of the optimization model (e.g., "DE", "GA", "Grid")
        cost_value: Final cost value
        n_evaluations: Number of evaluations performed
        optimization_time: Time taken for optimization
        sensitivities: Dictionary of parameter sensitivities
    
    Returns:
        None
    """
    # Create twin with best parameters
    I_scale, damping_coefficient, mass = best_params
    twin = create_twin_with_params(mass, I_scale, damping_coefficient)
    
    # Simulate with best parameters
    theta_sim = twin.simulate_passive(theta0, theta_dot0, time_array)
    
    # Calculate errors
    position_error, velocity_error, rms_error, max_error = calculate_errors(
        theta_sim, theta_real, calculate_velocity(theta_sim, time_array[1]-time_array[0]), theta_dot_real
    )
    
    # Calculate physical characteristics
    g = 9.81  # gravity
    L = twin.l  # pendulum length
    I = I_scale * mass * L**2
    
    # Natural frequency
    omega_n = np.sqrt(g/L)
    f_n = omega_n/(2*np.pi)
    
    # Damping ratio
    zeta = damping_coefficient/(2*np.sqrt(I*g/L))
    
    # Quality factor
    Q = 1/(2*zeta) if zeta > 0 else float('inf')
    
    # Create report content
    report = []
    report.append(f"PENDULUM OPTIMIZATION REPORT ({model_name})")
    report.append("=" * 50)
    report.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    report.append("OPTIMIZATION PARAMETERS")
    report.append("-" * 50)
    if model_name == "Grid":
        report.append(f"Grid Size: {n_evaluations}")
    elif model_name == "GA":
        report.append(f"Population Size: {n_evaluations // NUM_GENERATIONS if 'NUM_GENERATIONS' in globals() else 'N/A'}")
        report.append(f"Maximum Generations: {NUM_GENERATIONS if 'NUM_GENERATIONS' in globals() else 'N/A'}")
        report.append(f"Crossover Rate: {CROSSOVER_PROB if 'CROSSOVER_PROB' in globals() else 'N/A'}")
        report.append(f"Mutation Rate: {MUTATION_PROB if 'MUTATION_PROB' in globals() else 'N/A'}")
    elif model_name == "DE":
        report.append(f"Population Size: {POPULATION_SIZE if 'POPULATION_SIZE' in globals() else 'N/A'}")
        report.append(f"Maximum Iterations: {MAX_GENERATIONS if 'MAX_GENERATIONS' in globals() else 'N/A'}")
        report.append(f"Mutation Strategy: best1bin")
    
    report.append("")
    report.append("OPTIMIZATION RESULTS")
    report.append("-" * 50)
    report.append(f"Best Cost: {cost_value:.6f}")
    report.append(f"Number of Evaluations: {n_evaluations}")
    if optimization_time is not None:
        report.append(f"Optimization Time: {optimization_time:.2f} seconds")
    
    report.append("")
    report.append("BEST PARAMETERS")
    report.append("-" * 50)
    report.append(f"I_scale: {I_scale:.6f}")
    report.append(f"Damping Coefficient: {damping_coefficient:.6f}")
    report.append(f"Mass: {mass:.6f} kg")
    
    report.append("")
    report.append("PHYSICAL CHARACTERISTICS")
    report.append("-" * 50)
    report.append(f"Effective I: {I:.6f} kg⋅m²")
    report.append(f"Natural Frequency: {f_n:.4f} Hz")
    report.append(f"Damping Ratio: {zeta:.4f}")
    report.append(f"Quality Factor: {Q:.1f}")
    
    report.append("")
    report.append("ERROR METRICS")
    report.append("-" * 50)
    report.append(f"RMS Error: {rms_error:.6f} rad")
    report.append(f"Max Error: {max_error:.6f} rad")
    
    if sensitivities is not None:
        report.append("")
        report.append("PARAMETER SENSITIVITIES")
        report.append("-" * 50)
        report.append(f"I_scale Sensitivity: {sensitivities['I_scale']:.4f}")
        report.append(f"Damping Sensitivity: {sensitivities['damping']:.4f}")
        report.append(f"Mass Sensitivity: {sensitivities['mass']:.4f}")
    
    # Save report to file
    filename = f"reports/{title}_pendulum_{model_name}_report.txt" if title and model_name else "reports/optimization_report.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write('\n'.join(report))
    
    return report 