import time
import pygame
import numpy as np
from Digital_twin import DigitalTwin
import matplotlib.pyplot as plt
import os
import csv
from scipy.optimize import differential_evolution
import pandas as pd

class ParameterOptimizer:
    def __init__(self, target_file="datasets/filtered_datasets/move_a_17.7_kalman_output.csv"):
        self.digital_twin = DigitalTwin()
        self.target_file = target_file
        self.target_data = None
        self.load_target_data()
        
    def load_target_data(self):
        """Load the target movement data from CSV file"""
        try:
            self.target_data = pd.read_csv(self.target_file)
            print(f"Loaded target data from {self.target_file}")
        except Exception as e:
            print(f"Error loading target data: {e}")
            # Create dummy data if file doesn't exist
            self.target_data = pd.DataFrame({
                'time': np.linspace(0, 2, 100),
                'theta': np.sin(np.linspace(0, 4*np.pi, 100)) * 0.5,
                'theta_dot': np.cos(np.linspace(0, 4*np.pi, 100)) * 0.5,
                'x_pivot': np.sin(np.linspace(0, 2*np.pi, 100)) * 0.1
            })
            print("Created dummy target data for testing")
    
    def simulate_with_parameters(self, params):
        """Simulate the pendulum with given parameters and return error compared to target"""
        # Extract parameters
        a_m, key_duration = params
        
        # Set parameters in digital twin
        self.digital_twin.a_m = a_m
        
        # Reset pendulum state
        self.digital_twin.theta = 0.
        self.digital_twin.theta_dot = 0.
        self.digital_twin.x_pivot = 0.
        self.digital_twin.steps = 0.
        
        # Clear recording file
        os.makedirs('reports', exist_ok=True)
        with open('reports/recording.csv', mode='w', newline='') as file:
            file.truncate()
        
        # Simulate for 2 seconds
        simulation_time = 2.0
        steps = int(simulation_time / self.digital_twin.delta_t)
        
        # Record data
        recorded_data = []
        
        # Perform action at the beginning
        self.digital_twin.perform_action('left', key_duration)
        
        # Run simulation
        for _ in range(steps):
            theta, theta_dot, x_pivot, _ = self.digital_twin.step()
            current_time = _ * self.digital_twin.delta_t
            recorded_data.append([current_time, theta, theta_dot, x_pivot])
            
            # Save data
            with open('reports/recording.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([current_time, theta, theta_dot, x_pivot, self.digital_twin.currentmotor_acceleration])
        
        # Convert to DataFrame for easier comparison
        recorded_df = pd.DataFrame(recorded_data, columns=['time_sec', 'theta_kalman', 'theta_dot_kalman', 'x_pivot_m'])
        
        # Interpolate target data to match recorded data timestamps
        target_interp = pd.DataFrame()
        target_interp['time_sec'] = recorded_df['time_sec']
        target_interp['theta_kalman'] = np.interp(recorded_df['time_sec'], self.target_data['time_sec'], self.target_data['theta_kalman'])
        target_interp['theta_dot_kalman'] = np.interp(recorded_df['time_sec'], self.target_data['time_sec'], self.target_data['theta_dot_kalman'])
        target_interp['x_pivot_m'] = np.interp(recorded_df['time_sec'], self.target_data['time_sec'], self.target_data['x_pivot_m'])
        
        # Calculate error metrics
        theta_error = np.mean((recorded_df['theta_kalman'] - target_interp['theta_kalman'])**2)
        theta_dot_error = np.mean((recorded_df['theta_dot_kalman'] - target_interp['theta_dot_kalman'])**2)
        x_pivot_error = np.mean((recorded_df['x_pivot_m'] - target_interp['x_pivot_m'])**2)
        
        # Combined error (weighted)
        total_error = theta_error + 0.5 * theta_dot_error + 0.1 * x_pivot_error
        
        return total_error
    
    def objective_function(self, params):
        """Objective function for optimization"""
        try:
            error = self.simulate_with_parameters(params)
            print(f"Parameters: a_m={params[0]:.2f}, key_duration={params[1]:.0f}ms, Error: {error:.6f}")
            return error
        except Exception as e:
            print(f"Error in simulation: {e}")
            return 1e10  # Return large error to avoid this parameter set
    
    def optimize(self):
        """Run differential evolution to find optimal parameters"""
        # Define parameter bounds
        # a_m: [0.5, 1.5], key_duration: [100, 600] ms
        bounds = [(0.5, 1.5), (100, 600)]
        
        # Run differential evolution
        result = differential_evolution(
            self.objective_function,
            bounds,
            maxiter=20,
            popsize=10,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42
        )
        
        print("\nOptimization Results:")
        print(f"Optimal a_m: {result.x[0]:.4f}")
        print(f"Optimal key_duration: {result.x[1]:.0f}ms")
        print(f"Final error: {result.fun:.6f}")
        
        # Run simulation with optimal parameters
        self.simulate_with_parameters(result.x)
        
        # Plot results
        self.plot_results()
        
        return result.x
    
    def plot_results(self):
        """Plot the results of the simulation with optimal parameters"""
        # Load recorded data
        recorded_df = pd.read_csv('reports/recording.csv', header=None, 
                                 names=['time_sec', 'theta_kalman', 'theta_dot_kalman', 'x_pivot_m', 'acceleration'])
        
        # Create figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)
        
        # Plot theta
        axs[0].plot(recorded_df['time_sec'], recorded_df['theta_kalman'], label='Simulated', color='blue')
        axs[0].plot(self.target_data['time_sec'], self.target_data['theta_kalman'], label='Target', color='red', linestyle='--')
        axs[0].set_ylabel('θ (radians)')
        axs[0].set_title('Pendulum Angle')
        axs[0].legend()
        axs[0].grid()
        
        # Plot theta_dot
        axs[1].plot(recorded_df['time_sec'], recorded_df['theta_dot_kalman'], label='Simulated', color='green')
        axs[1].plot(self.target_data['time_sec'], self.target_data['theta_dot_kalman'], label='Target', color='orange', linestyle='--')
        axs[1].set_ylabel('θ̇ (rad/s)')
        axs[1].set_title('Angular Velocity')
        axs[1].legend()
        axs[1].grid()
        
        # Plot x_pivot
        axs[2].plot(recorded_df['time_sec'], recorded_df['x_pivot_m'], label='Simulated', color='purple')
        axs[2].plot(self.target_data['time_sec'], self.target_data['x_pivot_m'], label='Target', color='brown', linestyle='--')
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('x_pivot (m)')
        axs[2].set_title('Cart Position')
        axs[2].legend()
        axs[2].grid()
        
        plt.suptitle('Comparison of Simulated vs Target Movement')
        plt.savefig('reports/optimization_results.png')
        plt.show()

if __name__ == "__main__":
    # Create optimizer
    optimizer = ParameterOptimizer()
    
    # Run optimization
    optimal_params = optimizer.optimize()
    
    print("\nOptimal parameters found:")
    print(f"a_m = {optimal_params[0]:.4f}")
    print(f"key_duration = {optimal_params[1]:.0f}ms") 