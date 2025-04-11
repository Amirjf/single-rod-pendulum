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
from utility import create_twin_with_params, load_real_data

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

def objective_function(params, real_data):
    """
    Simplified objective function to minimize the difference between simulated and real motion.
    Only compares data after the 'A' button press.
    """
    a_m, duration = params
    print(f"\n{'='*50}")
    print(f"Evaluating parameters: a_m={a_m:.3f}, duration={duration:.3f}")
    print(f"{'='*50}")
    
    # Create a new twin instance for each evaluation with Pygame disabled
    mass = 0.527869
    I_scale = 0.705700
    damping_coefficient = 0.008006
    twin = create_twin_with_params(mass, I_scale, damping_coefficient, use_pygame=False)
    
    # Set motor parameters
    twin.a_m = a_m
    twin.duration_of_action_a = duration
    
    # Find when motor action starts in real data
    x_pivot = real_data['x_pivot_m'].values
    action_start_idx = np.where(np.abs(x_pivot) > 0.0001)[0][0]
    action_start_time = real_data['time_sec'].iloc[action_start_idx]
    print(f"Action starts at t={action_start_time:.3f}s")
    
    # Get initial conditions from real data BEFORE the action
    pre_action_idx = action_start_idx - 1
    twin.theta = real_data['theta_kalman'].iloc[pre_action_idx]
    twin.theta_dot = real_data['theta_dot_kalman'].iloc[pre_action_idx]
    twin.t = real_data['time_sec'].iloc[pre_action_idx]
    print(f"Initial conditions: theta={twin.theta:.3f}, theta_dot={twin.theta_dot:.3f}")
    
    # Update motor accelerations for the action
    twin.update_motor_accelerations_real('left', duration)
    
    # Simulate motion with larger time steps
    simulated_theta = []
    simulated_theta_dot = []
    simulated_x = []
    simulated_times = []
    
    # Simulate for the same duration as real data
    end_time = real_data['time_sec'].iloc[-1]
    steps = 0
    last_progress = -1
    
    print("\nSimulation Progress:")
    print("Time (s) | Steps | Theta | Theta_dot | X_pivot")
    print("-" * 60)
    
    while twin.t <= end_time:
        twin.step()
        simulated_theta.append(twin.theta)
        simulated_theta_dot.append(twin.theta_dot)
        simulated_x.append(twin.x_pivot)
        simulated_times.append(twin.t)
        steps += 1
        
        # Print progress every 0.1 seconds
        progress = int(twin.t * 10)
        if progress > last_progress:
            print(f"{twin.t:6.2f}s | {steps:5d} | {twin.theta:6.3f} | {twin.theta_dot:8.3f} | {twin.x_pivot:8.3f}")
            last_progress = progress
    
    print(f"\nSimulation completed: {steps} steps, final time: {twin.t:.3f}s")
    
    # Convert to numpy arrays
    simulated_theta = np.array(simulated_theta)
    simulated_theta_dot = np.array(simulated_theta_dot)
    simulated_x = np.array(simulated_x)
    simulated_times = np.array(simulated_times)
    
    # Find indices where simulated time matches real time after action start
    mask = simulated_times >= action_start_time
    simulated_theta = simulated_theta[mask]
    simulated_theta_dot = simulated_theta_dot[mask]
    simulated_x = simulated_x[mask]
    
    real_theta = real_data['theta_kalman'].values[action_start_idx:]
    real_theta_dot = real_data['theta_dot_kalman'].values[action_start_idx:]
    real_x = real_data['x_pivot_m'].values[action_start_idx:]
    
    # Ensure same length
    min_len = min(len(simulated_theta), len(real_theta))
    simulated_theta = simulated_theta[:min_len]
    simulated_theta_dot = simulated_theta_dot[:min_len]
    simulated_x = simulated_x[:min_len]
    real_theta = real_theta[:min_len]
    real_theta_dot = real_theta_dot[:min_len]
    real_x = real_x[:min_len]
    
    # Simplified cost calculation
    theta_cost = np.mean((simulated_theta - real_theta) ** 2)
    theta_dot_cost = np.mean((simulated_theta_dot - real_theta_dot) ** 2)
    x_cost = np.mean((simulated_x - real_x) ** 2)
    
    # Weight the costs
    total_cost = theta_cost + 0.1 * theta_dot_cost + 0.01 * x_cost
    
    print("\nCost Analysis:")
    print(f"Theta cost: {theta_cost:.6f}")
    print(f"Theta dot cost: {theta_dot_cost:.6f}")
    print(f"X pivot cost: {x_cost:.6f}")
    print(f"Total cost: {total_cost:.6f}")
    print(f"{'='*50}\n")
    
    return total_cost

def optimize_motor_params():
    """Run optimization to find best motor parameters with reduced iterations"""
    print("\nStarting motor parameter optimization...")
    
    # Load real data
    real_data = pd.read_csv('datasets/filtered_datasets/move_a_17.7_kalman_output.csv')
    print(f"Loaded {len(real_data)} data points")
    print(f"Time range: {real_data['time_sec'].min():.3f}s to {real_data['time_sec'].max():.3f}s")
    
    # Define bounds for parameters
    bounds = [
        (0.1, 10.0),    # a_m: motor acceleration
        (0.3, 0.6)      # duration_of_action_a: duration in seconds
    ]
    print(f"Parameter bounds: a_m={bounds[0]}, duration={bounds[1]}")
    
    # Create a pool of workers
    num_cores = multiprocessing.cpu_count()
    print(f"\nUsing {num_cores} CPU cores for parallel optimization")
    
    # Create a partial function with fixed arguments
    obj_func = partial(objective_function, real_data=real_data)
    
    print("\nStarting differential evolution optimization...")
    print("This may take a while. Progress will be shown for each evaluation.")
    
    # Run differential evolution optimization with reduced iterations
    result = differential_evolution(
        obj_func,
        bounds=bounds,
        maxiter=50,  # Reduced from 100
        popsize=10,  # Reduced from 20
        tol=0.01,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        workers=num_cores,
        updating='deferred',
        callback=lambda x, convergence: print(f"\nIteration complete. Best parameters so far: a_m={x[0]:.3f}, duration={x[1]:.3f}, cost={convergence:.6f}")
    )
    
    # Print optimization results
    print("\nOptimization Results:")
    print(f"Best a_m: {result.x[0]:.6f}")
    print(f"Best duration: {result.x[1]:.6f} seconds")
    print(f"Best cost: {result.fun:.6f}")
    print(f"Number of evaluations: {result.nfev}")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    
    # Simulate and plot with best parameters
    print("\nSimulating with best parameters...")
    simulated_theta, simulated_theta_dot, a_m, duration = simulate_and_plot(result.x)
    
    return result.x

def simulate_and_plot(params):
    """Simulate and plot results with given parameters"""
    # Load real data
    real_data = pd.read_csv('datasets/filtered_datasets/move_a_17.7_kalman_output.csv')
    
    # Find when motor action starts in real data
    x_pivot = real_data['x_pivot_m'].values
    action_start_idx = np.where(np.abs(x_pivot) > 0.0001)[0][0]  # First non-zero position
    action_start_time = real_data['time_sec'].iloc[action_start_idx]
    
    # Create digital twin with initial parameters
    mass = 0.527869  # From differential evolution optimization
    I_scale = 0.705700  # From differential evolution optimization
    damping_coefficient = 0.008006  # From differential evolution optimization
    
    # Use Pygame for the final visualization
    twin = create_twin_with_params(mass, I_scale, damping_coefficient, use_pygame=True)
    
    # Set motor parameters
    a_m, duration = params
    twin.a_m = a_m
    twin.duration_of_action_a = duration
    
    # Get initial conditions from real data BEFORE the action
    pre_action_idx = action_start_idx - 1  # Use the last point before action
    twin.theta = real_data['theta_kalman'].iloc[pre_action_idx]
    twin.theta_dot = real_data['theta_dot_kalman'].iloc[pre_action_idx]
    twin.t = real_data['time_sec'].iloc[pre_action_idx]
    
    # Update motor accelerations for the action
    twin.update_motor_accelerations_real('left', duration)  # Duration is already in seconds
    
    # Simulate motion with best parameters
    simulated_theta = []
    simulated_theta_dot = []
    simulated_x = []
    times = []
    
    # Simulate for the same duration as real data
    end_time = real_data['time_sec'].iloc[-1]
    while twin.t <= end_time:
        # Update twin state
        twin.step()
        
        # Store simulated values
        simulated_theta.append(twin.theta)
        simulated_theta_dot.append(twin.theta_dot)
        simulated_x.append(twin.x_pivot)
        times.append(twin.t)
    
    # Create plots
    plt.figure(figsize=(15, 10))
    
    # Plot theta
    plt.subplot(3, 1, 1)
    plt.plot(real_data['time_sec'], real_data['theta_kalman'], 'b-', label='Real')
    plt.plot(times, simulated_theta, 'r--', label='Simulated')
    plt.axvline(x=action_start_time, color='g', linestyle='--', label='Action Start')
    plt.xlabel('Time (s)')
    plt.ylabel('θ (rad)')
    plt.title('Angle Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot theta_dot
    plt.subplot(3, 1, 2)
    plt.plot(real_data['time_sec'], real_data['theta_dot_kalman'], 'b-', label='Real')
    plt.plot(times, simulated_theta_dot, 'r--', label='Simulated')
    plt.axvline(x=action_start_time, color='g', linestyle='--', label='Action Start')
    plt.xlabel('Time (s)')
    plt.ylabel('θ̇ (rad/s)')
    plt.title('Angular Velocity Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot x_pivot
    plt.subplot(3, 1, 3)
    plt.plot(real_data['time_sec'], real_data['x_pivot_m'], 'b-', label='Real')
    plt.plot(times, simulated_x, 'r--', label='Simulated')
    plt.axvline(x=action_start_time, color='g', linestyle='--', label='Action Start')
    plt.xlabel('Time (s)')
    plt.ylabel('x_pivot (m)')
    plt.title('Pivot Displacement Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('optimization_results.png')
    plt.show()
    
    return simulated_theta, simulated_theta_dot, a_m, duration

def plot_comparison(real_data, simulated_data, action_start_time):
    """Plot comparison between real and simulated data"""
    plt.figure(figsize=(15, 10))
    
    # Plot theta
    plt.subplot(3, 1, 1)
    plt.plot(real_data['time_sec'], real_data['theta_kalman'], 'b-', label='Real', alpha=0.7)
    plt.plot(simulated_data['times'], simulated_data['theta'], 'r--', label='Simulated', alpha=0.7)
    plt.axvline(x=action_start_time, color='g', linestyle='--', label='Action Start')
    plt.xlabel('Time (s)')
    plt.ylabel('θ (rad)')
    plt.title('Angle Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot theta_dot
    plt.subplot(3, 1, 2)
    plt.plot(real_data['time_sec'], real_data['theta_dot_kalman'], 'b-', label='Real', alpha=0.7)
    plt.plot(simulated_data['times'], simulated_data['theta_dot'], 'r--', label='Simulated', alpha=0.7)
    plt.axvline(x=action_start_time, color='g', linestyle='--', label='Action Start')
    plt.xlabel('Time (s)')
    plt.ylabel('θ̇ (rad/s)')
    plt.title('Angular Velocity Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot x_pivot
    plt.subplot(3, 1, 3)
    plt.plot(real_data['time_sec'], real_data['x_pivot_m'], 'b-', label='Real', alpha=0.7)
    plt.plot(simulated_data['times'], simulated_data['x'], 'r--', label='Simulated', alpha=0.7)
    plt.axvline(x=action_start_time, color='g', linestyle='--', label='Action Start')
    plt.xlabel('Time (s)')
    plt.ylabel('x_pivot (m)')
    plt.title('Pivot Displacement Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('optimization_results.png')
    plt.close()

# Main execution
if __name__ == "__main__":
    result = optimize_motor_params() 