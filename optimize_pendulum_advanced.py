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

class ModifiedDigitalTwin(DigitalTwin):
    """Extended version of DigitalTwin with more flexible physics model"""
    
    def __init__(self):
        super().__init__()
        self.c_angle = 0.0  # Position-dependent friction coefficient
    
    def get_theta_double_dot(self, theta, theta_dot):
        """
        Modified physics model with improved frequency and damping
        """
        # Simplified moment of inertia
        I_total = self.I
        
        # Basic gravity torque
        torque_gravity = -(self.mp * self.g * self.l / I_total) * np.sin(theta)
        
        # Velocity-dependent air friction (quadratic at higher speeds)
        torque_air_friction = -(self.c_air / I_total) * theta_dot * abs(theta_dot)
        
        # Position and velocity dependent Coulomb friction
        # More friction at larger angles and higher speeds
        angle_factor = 1.0 + self.c_angle * theta**2
        torque_coulomb_friction = -(self.c_c / I_total) * angle_factor * theta_dot
        
        return torque_gravity + torque_air_friction + torque_coulomb_friction

# Load real data
csv_filename = "half_theta_2"

def load_real_data(filename=None, start_time=1.0):
    if filename is None:
        filename = f"datasets/filtered_datasets/{csv_filename}_kalman_output.csv"
    
    df = pd.read_csv(filename)
    df_trimmed = df[df['time_sec'] >= start_time].reset_index(drop=True)
    
    time_array = df_trimmed['time_sec'].values
    theta_real = df_trimmed['theta_kalman'].values
    theta_dot_real = df_trimmed['theta_dot_kalman'].values
    
    # Get initial conditions
    theta0 = theta_real[0]
    theta_dot0 = theta_dot_real[0]
    
    print(f"Initial conditions: theta0 = {theta0:.4f}, theta_dot0 = {theta_dot0:.4f}")
    return time_array, theta_real, theta_dot_real, theta0, theta_dot0

def parallel_cost_function(params, time_array, theta_real, theta_dot_real, theta0, theta_dot0):
    """Modified cost function including mass optimization"""
    # Unpack parameters - now including mass
    I_scale, c_air, c_c, c_angle, mass = params
    
    # Create modified twin with these parameters
    twin = ModifiedDigitalTwin()
    twin.mp = mass        # Mass is now a parameter
    twin.l = 0.35        # Fixed length (35cm)
    twin.I = I_scale * twin.mp * twin.l**2
    twin.c_air = c_air
    twin.c_c = c_c
    twin.c_angle = c_angle
    twin.currentmotor_acceleration = 0
    
    # Simulate with these parameters
    theta_sim = twin.simulate_passive(theta0, theta_dot0, time_array)
    
    # Check for invalid simulation
    if np.any(np.isnan(theta_sim)) or np.any(np.isinf(theta_sim)):
        return 1e6
    
    min_len = min(len(theta_sim), len(theta_real))
    dt = time_array[1] - time_array[0]
    
    # Calculate velocity for simulation
    theta_dot_sim = np.gradient(theta_sim[:min_len], dt)
    
    # 1. Time-domain position error with exponential weighting
    time_weights = np.exp(np.linspace(0, 1, min_len))
    time_domain_error = np.mean(time_weights * (theta_sim[:min_len] - theta_real[:min_len])**2)
    
    # 2. Velocity matching error
    velocity_error = np.mean((theta_dot_sim - theta_dot_real[:min_len])**2)
    
    # 3. Energy calculation and matching
    g = 9.81  # gravity
    l = 0.35  # pendulum length
    
    # Calculate energies for real data
    KE_real = 0.5 * mass * l**2 * theta_dot_real[:min_len]**2
    PE_real = mass * g * l * (1 - np.cos(theta_real[:min_len]))
    E_real = KE_real + PE_real
    
    # Calculate energies for simulated data
    KE_sim = 0.5 * mass * l**2 * theta_dot_sim**2
    PE_sim = mass * g * l * (1 - np.cos(theta_sim[:min_len]))
    E_sim = KE_sim + PE_sim
    
    # Energy matching error
    energy_error = np.mean((E_sim - E_real)**2)
    
    # 4. Frequency analysis
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
    
    # 5. Peak amplitude analysis
    real_peaks = signal.find_peaks(np.abs(theta_real[:min_len]), distance=15)[0]
    sim_peaks = signal.find_peaks(np.abs(theta_sim[:min_len]), distance=15)[0]
    
    if len(real_peaks) < 3 or len(sim_peaks) < 3:
        return 1e6
    
    n_peaks = min(len(real_peaks), len(sim_peaks))
    real_amplitudes = np.abs(theta_real[real_peaks[:n_peaks]])
    sim_amplitudes = np.abs(theta_sim[sim_peaks[:n_peaks]])
    
    # Amplitude error with decay rate
    amplitude_error = np.mean((real_amplitudes - sim_amplitudes)**2)
    
    # Calculate amplitude decay rates
    real_decay = real_amplitudes[1:] / real_amplitudes[:-1]
    sim_decay = sim_amplitudes[1:] / sim_amplitudes[:-1]
    decay_error = np.mean((real_decay - sim_decay)**2)
    
    # Combined weighted error with new terms
    total_error = (
        50 * time_domain_error +    # Base position matching
        200 * freq_error +          # Strong emphasis on frequency
        150 * amplitude_error +     # Good weight on amplitude
        100 * velocity_error +      # New: velocity matching
        50 * energy_error +         # New: energy conservation
        50 * decay_error           # New: explicit decay rate matching
    )
    
    return total_error

def optimize_pendulum_params():
    print("Starting enhanced parallel pendulum parameter optimization...")
    start_time = time.time()
    
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} CPU cores for parallel processing")
    
    cost_func = partial(parallel_cost_function, 
                       time_array=time_array,
                       theta_real=theta_real,
                       theta_dot_real=theta_dot_real,
                       theta0=theta0,
                       theta_dot0=theta_dot0)
    
    # Modified bounds including mass
    bounds = [
        (0.80, 0.90),      # I_scale: For frequency matching
        (0.00001, 0.001),  # c_air: For quadratic air friction
        (0.00001, 0.01),   # c_c: For Coulomb friction
        (0.0, 1.0),        # c_angle: For position-dependent effects
        (0.8, 1.2)         # mass: Allow ±20% variation around nominal 1.0
    ]
    
    # Enhanced optimization settings
    result = differential_evolution(
        cost_func, 
        bounds, 
        popsize=40,
        maxiter=300,
        tol=0.000001,
        mutation=(0.5, 1.5),
        recombination=0.9,
        disp=True,
        updating='deferred',
        workers=num_cores
    )
    
    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time:.1f} seconds")
    return result

# Modify the plot_frequency_analysis function to return its axis instead of showing
def plot_frequency_analysis(theta_real, theta_sim, time_array, ax=None):
    min_len = min(len(theta_sim), len(theta_real))
    dt = time_array[1] - time_array[0]  # Time step
    
    # Compute FFTs
    n_points = 2048  # Higher for better resolution
    real_fft = fft(theta_real[:min_len], n_points)
    sim_fft = fft(theta_sim[:min_len], n_points)
    
    # Frequency axis
    freq = np.fft.fftfreq(n_points, d=dt)[:n_points//2]
    
    # Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot only the positive frequencies (up to Nyquist frequency)
    ax.plot(freq, np.abs(real_fft[:n_points//2]), 'b-', label='Real Data')
    ax.plot(freq, np.abs(sim_fft[:n_points//2]), 'r--', label='Simulation')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.set_title('Frequency Domain Analysis')
    ax.grid(True)
    ax.legend()
    
    # Zoom in on the lower frequencies (the most important part)
    ax.set_xlim(0, 2)  # Focus on 0-2 Hz range
    
    return ax

def add_parameter_box(fig, params, twin, rms_error, max_error):
    """Add parameter information box to the figure"""
    I_scale, c_air, c_c, c_angle, mass = params
    param_text = (
        f'Model Parameters:\n'
        f'I_scale = {I_scale:.6f}\n'
        f'c_air = {c_air:.6f}\n'
        f'c_c = {c_c:.6f}\n'
        f'c_angle = {c_angle:.6f}\n'
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

# Add a new comprehensive plot function
def plot_comprehensive_analysis(theta_real, theta_sim, time_array, theta_dot_real=None):
    """Create comprehensive analysis plots including additional metrics"""
    min_len = min(len(theta_sim), len(theta_real))
    dt = time_array[1] - time_array[0]
    
    # Calculate theta_dot for simulation
    theta_dot_sim = np.gradient(theta_sim, dt)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 20))
    
    # 1. Time-domain position comparison (Original)
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
    
    # 4. Energy profiles
    g = 9.81
    l = 0.35  # pendulum length
    m = 1.0   # mass
    
    # Calculate energies
    KE_real = 0.5 * m * l**2 * theta_dot_real[:min_len]**2
    PE_real = m * g * l * (1 - np.cos(theta_real[:min_len]))
    E_real = KE_real + PE_real
    
    KE_sim = 0.5 * m * l**2 * theta_dot_sim[:min_len]**2
    PE_sim = m * g * l * (1 - np.cos(theta_sim[:min_len]))
    E_sim = KE_sim + PE_sim
    
    ax4 = fig.add_subplot(4, 2, 4)
    ax4.plot(time_array[:min_len], E_real, 'b-', label='Real Energy')
    ax4.plot(time_array[:min_len], E_sim, 'r--', label='Simulated Energy')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Energy (J)')
    ax4.set_title('Total Energy Over Time')
    ax4.grid(True)
    ax4.legend()
    
    # 5. Zero-crossing and Peak timing
    real_zeros = np.where(np.diff(np.signbit(theta_real[:min_len])))[0]
    sim_zeros = np.where(np.diff(np.signbit(theta_sim[:min_len])))[0]
    
    real_peaks, _ = signal.find_peaks(np.abs(theta_real[:min_len]))
    sim_peaks, _ = signal.find_peaks(np.abs(theta_sim[:min_len]))
    
    ax5 = fig.add_subplot(4, 2, 5)
    ax5.plot(time_array[:min_len], theta_real[:min_len], 'b-', alpha=0.5)
    ax5.plot(time_array[:min_len], theta_sim[:min_len], 'r--', alpha=0.5)
    ax5.plot(time_array[real_zeros], np.zeros_like(real_zeros), 'bo', label='Real Zeros')
    ax5.plot(time_array[sim_zeros], np.zeros_like(sim_zeros), 'ro', label='Sim Zeros')
    ax5.plot(time_array[real_peaks], theta_real[real_peaks], 'bx', label='Real Peaks')
    ax5.plot(time_array[sim_peaks], theta_sim[sim_peaks], 'rx', label='Sim Peaks')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Angle (rad)')
    ax5.set_title('Zero Crossings and Peaks')
    ax5.grid(True)
    ax5.legend()
    
    # 6. Local frequency variation
    from scipy.signal import hilbert
    analytic_real = hilbert(theta_real[:min_len])
    analytic_sim = hilbert(theta_sim[:min_len])
    
    inst_freq_real = np.diff(np.unwrap(np.angle(analytic_real))) / (2.0*np.pi*dt)
    inst_freq_sim = np.diff(np.unwrap(np.angle(analytic_sim))) / (2.0*np.pi*dt)
    
    ax6 = fig.add_subplot(4, 2, 6)
    ax6.plot(time_array[1:min_len], inst_freq_real, 'b-', label='Real')
    ax6.plot(time_array[1:min_len], inst_freq_sim, 'r--', label='Simulated')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Frequency (Hz)')
    ax6.set_title('Instantaneous Frequency')
    ax6.grid(True)
    ax6.legend()
    
    # 7. Error distributions
    position_error = theta_sim[:min_len] - theta_real[:min_len]
    velocity_error = theta_dot_sim[:min_len] - theta_dot_real[:min_len]
    
    ax7 = fig.add_subplot(4, 2, 7)
    ax7.hist(position_error, bins=50, alpha=0.5, label='Position Error')
    ax7.hist(velocity_error, bins=50, alpha=0.5, label='Velocity Error')
    ax7.set_xlabel('Error')
    ax7.set_ylabel('Count')
    ax7.set_title('Error Distributions')
    ax7.grid(True)
    ax7.legend()
    
    # 8. Original FFT analysis
    ax8 = fig.add_subplot(4, 2, 8)
    n_points = 8192
    freq = np.fft.fftfreq(n_points, d=dt)[:n_points//2]
    real_fft = np.abs(fft(theta_real[:min_len], n_points)[:n_points//2])
    sim_fft = np.abs(fft(theta_sim[:min_len], n_points)[:n_points//2])
    ax8.plot(freq, real_fft, 'b-', label='Real Data')
    ax8.plot(freq, sim_fft, 'r--', label='Simulation')
    ax8.set_xlabel('Frequency (Hz)')
    ax8.set_ylabel('Magnitude')
    ax8.set_title('Frequency Domain Analysis')
    ax8.set_xlim(0, 2)
    ax8.grid(True)
    ax8.legend()
    
    plt.tight_layout()
    return fig

def simulate_and_plot(params):
    """Simulate and create plots with given parameters"""
    I_scale, c_air, c_c, c_angle, mass = params
    
    # Create twin with parameters
    twin = ModifiedDigitalTwin()
    twin.mp = mass
    twin.l = 0.35
    twin.I = I_scale * twin.mp * twin.l**2
    twin.c_air = c_air
    twin.c_c = c_c
    twin.c_angle = c_angle
    twin.currentmotor_acceleration = 0
    
    # Simulate
    theta_sim = twin.simulate_passive(theta0, theta_dot0, time_array)
    
    # Calculate velocity from position data
    dt = time_array[1] - time_array[0]
    theta_dot_sim = np.gradient(theta_sim, dt)
    theta_dot_real = np.gradient(theta_real, dt)  # Calculate velocity from real position data
    
    # Create plots
    fig = plot_comprehensive_analysis(theta_real, theta_sim, time_array, theta_dot_real)
    
    # Calculate metrics
    min_len = min(len(theta_sim), len(theta_real))
    error = theta_sim[:min_len] - theta_real[:min_len]
    rms_error = np.sqrt(np.mean(error**2))
    max_error = np.max(np.abs(error))
    
    # Add parameter box
    add_parameter_box(fig, params, twin, rms_error, max_error)
    
    plt.savefig('reports/{csv_filename}_pendulum_GA_analysis.png', dpi=300)
    plt.show()
    
    # Print detailed results
    print("\nFinal Parameters and Physics Values:")
    print("-" * 50)
    print(f"Mass (mp): {mass:.4f} kg")
    print(f"Length (l): {twin.l:.4f} m")
    print(f"Moment of Inertia Scale: {I_scale:.4f}")
    print(f"Effective I: {twin.I:.6f} kg⋅m²")
    print(f"Air Friction (c_air): {c_air:.8f}")
    print(f"Coulomb Friction (c_c): {c_c:.8f}")
    print(f"Angle-Dependent Friction (c_angle): {c_angle:.8f}")
    
    print("\nPerformance Metrics:")
    print("-" * 50)
    print(f"RMS Error: {rms_error:.4f} rad")
    print(f"Max Absolute Error: {max_error:.4f} rad")
    
    # Frequency analysis
    zero_crossings = np.where(np.diff(np.signbit(theta_real[:min_len])))[0]
    
    if len(zero_crossings) >= 4:
        periods = np.diff(zero_crossings)[::2] * 2
        avg_period = np.mean(periods) * (time_array[1] - time_array[0])
        natural_freq = 1 / avg_period
        theoretical_freq = np.sqrt(9.8065/(twin.l))/(2*np.pi)
        
        print("\nFrequency Analysis:")
        print("-" * 50)
        print(f"Measured Natural Frequency: {natural_freq:.4f} Hz")
        print(f"Theoretical Natural Frequency: {theoretical_freq:.4f} Hz")
        print(f"Frequency Error: {abs(natural_freq - theoretical_freq):.4f} Hz")
    
    return theta_sim, error, I_scale, c_air, c_c, c_angle, mass

# Main execution
if __name__ == "__main__":
    # Load the real data
    time_array, theta_real, theta_dot_real, theta0, theta_dot0 = load_real_data()
    
    # Run the optimization
    result = optimize_pendulum_params()
    
    # Get the best parameters
    I_scale, c_air, c_c, c_angle, mass = result.x
    
    # Simulate and plot with best parameters
    theta_sim, error, I_scale, c_air, c_c, c_angle, mass = simulate_and_plot(result.x)
    
    # Print optimization summary
    print("\nOptimization Summary:")
    print("-" * 50)
    print(f"Success: {result.success}")
    print(f"Final cost value: {result.fun:.6f}")
    print(f"Number of evaluations: {result.nfev}")
    print(f"Number of iterations: {result.nit}")
    
    # Enhanced sensitivity analysis
    print("\nSensitivity Analysis:")
    print("-" * 50)
    print("Testing I_scale variations:")
    for i in range(-2, 3):
        if i == 0:
            continue
        factor = 1 + i*0.1
        test_I_scale = I_scale * factor
        test_params = [test_I_scale, c_air, c_c, c_angle, mass]
        test_cost = parallel_cost_function(test_params, time_array, theta_real, theta_dot_real, theta0, theta_dot0)
        print(f"I_scale={test_I_scale:.4f} (change: {i*10:+d}%) → Cost={test_cost:.6f}")
    
    print("\nTesting mass variations:")
    for i in range(-2, 3):
        if i == 0:
            continue
        factor = 1 + i*0.1
        test_mass = mass * factor
        test_params = [I_scale, c_air, c_c, c_angle, test_mass]
        test_cost = parallel_cost_function(test_params, time_array, theta_real, theta_dot_real, theta0, theta_dot0)
        print(f"mass={test_mass:.4f} (change: {i*10:+d}%) → Cost={test_cost:.6f}")

    # The frequency analysis is already included in the comprehensive plot
    # plot_frequency_analysis(theta_real, theta_sim, time_array) 