import time
import pygame
from Digital_twin import DigitalTwin
import csv
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import numpy as np

motor_data_path = "reports/motor_data.csv"

def simulate_pendulum(automated=False, best_sequence=None):
    """
    Simulate the pendulum with either automated or manual control.
    
    Args:
        automated (bool): If True, runs in automated mode using best_sequence
        best_sequence (list): List of (time, action) tuples for automated mode
    """
    # Clear the contents of the recording.csv file
    with open('reports/recording.csv', mode='w', newline='') as file:
        file.truncate()

    digital_twin = DigitalTwin()
    running = True
    last_action = "None"  # Track the last action performed
    start_time = time.time()

    while running:
        current_time = time.time() - start_time
        
        # Step through simulation
        theta, theta_dot, x_pivot, currentmotor_acceleration = digital_twin.step()

        # Render with updated information
        digital_twin.render(theta, x_pivot, last_action)

        # Sleep for time step
        time.sleep(digital_twin.delta_t)

        # Save the data to CSV
        with open('reports/recording.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([current_time, theta, theta_dot, x_pivot, currentmotor_acceleration])

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if automated:
                    # In automated mode, only handle quit and escape
                    if event.key == pygame.K_ESCAPE:
                        running = False
                else:
                    # Manual control mode
                    if event.key in digital_twin.actions:
                        direction, duration = digital_twin.actions[event.key]
                        digital_twin.perform_action(direction, duration)
                        last_action = f"{direction.capitalize()} ({duration} ms)"
                    elif event.key == pygame.K_r:
                        digital_twin = DigitalTwin()  # Restart the system
                        last_action = "System Restarted"
                        print("System restarted")
                    elif event.key == pygame.K_b:  # 'b' for best sequence
                        print("Executing best sequence...")
                        if not execute_best_sequence(digital_twin):
                            running = False
                    elif event.key == pygame.K_ESCAPE:
                        running = False

        # In automated mode, check if it's time for the next action
        if automated and best_sequence:
            for target_time, action in best_sequence:
                if abs(current_time - target_time) < digital_twin.delta_t:
                    direction, duration = action
                    digital_twin.perform_action(direction, duration)
                    last_action = f"{direction.capitalize()} ({duration} ms)"
                    print(f"Automated action: {direction} ({duration}ms) at {current_time:.2f}s")

    pygame.quit()

def execute_best_sequence(digital_twin):
    """Execute the best action sequence found by the genetic algorithm."""
    # Best sequence timing and actions using ASCII key values
    sequence = [
        (0.0, ('left', 450)),    # Time 0.0s: Left (450ms)
        (0.45, ('right',350)),  # Time 0.45s: Right (350ms)
        (0.80, ('left', 450)),   # Time 0.80s: Left (450ms)
        (1.25, ('left', 350)),   # Time 1.25s: Left (350ms)
    ]
    
    start_time = time.time()
    last_action_time = -1
    
    print("Starting sequence execution...")
    print(f"Available actions: {digital_twin.actions}")
    
    for target_time, (direction, duration) in sequence:
        current_time = time.time() - start_time
        
        # Wait until it's time for the next action
        while current_time < target_time:
            # Step the simulation
            theta, theta_dot, x_pivot, currentmotor_acceleration = digital_twin.step()
            
            # Render current state
            digital_twin.render(theta, x_pivot, f"Next action at {target_time:.1f}s (in {(target_time-current_time):.1f}s)")
            
            # Save data
            with open('reports/recording.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([time.time(), theta, theta_dot, x_pivot, currentmotor_acceleration])
            
            # Sleep for time step
            time.sleep(digital_twin.delta_t)
            
            # Update current time
            current_time = time.time() - start_time
            
            # Check for quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return False
        
        # Perform the action
        digital_twin.perform_action(direction, duration)
        last_action_time = current_time
        print(f"Executing action: {direction} ({duration}ms) at time {current_time:.1f}s")
    
    print("Sequence execution completed")
    return True

def plot_motor_dynamics(csv_file, save_path="reports/motor_Acc_vel_angle.png"):
    df = pd.read_csv(csv_file)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)

    # Motor Acceleration
    axs[0].plot(df["time_s"], df["alpha_m_rad_s2"], label="Motor Acceleration (α_m)", color='blue')
    axs[0].set_ylabel("α_m (rad/s²)")
    axs[0].set_title("Motor Acceleration Over Time")
    axs[0].legend()
    axs[0].grid()

    # Motor Velocity
    axs[1].plot(df["time_s"], df["omega_m_rad_s"], label="Motor Velocity (ω_m)", color='red')
    axs[1].set_ylabel("ω_m (rad/s)")
    axs[1].set_title("Motor Velocity Over Time") 
    axs[1].legend()
    axs[1].grid()

    # Motor Position
    axs[2].plot(df["time_s"], df["theta_m_rad"], label="Motor Angle (θ_m)", color='green')
    axs[2].set_ylabel("θ_m (radians)")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_title("Motor Angle Displacement Over Time")
    axs[2].legend()
    axs[2].grid()

    plt.suptitle("Motor Acceleration, Velocity, and Angle")
    plt.savefig(save_path)
    plt.show()

def plot_system_state(csv_file, save_path="reports/state_space_analysis.png"):
    state_df = pd.read_csv(csv_file, header=None, names=["time", "theta", "theta_dot", "x_pivot", "acceleration"])

    # Debugging: Print min/max values for x_pivot
    print(f"x_pivot Min: {state_df['x_pivot'].min()}, Max: {state_df['x_pivot'].max()}")

    fig, axs = plt.subplots(4, 1, figsize=(10, 8), constrained_layout=True)

    # Pendulum Angle
    axs[0].plot(state_df["time"], state_df["theta"], label="Angle (θ)", color='blue')
    axs[0].set_ylabel("θ (radians)")
    axs[0].set_title("Pendulum Angle Over Time")
    axs[0].legend()
    axs[0].grid()

    # Angular Velocity
    axs[1].plot(state_df["time"], state_df["theta_dot"], label="Angular Velocity (θ̇)", color='red')
    axs[1].set_ylabel("θ̇ (rad/s)")
    axs[1].set_title("Pendulum Angular Velocity Over Time") 
    axs[1].legend()
    axs[1].grid()

    # State-Space Trajectory (Theta vs. Theta_dot)
    axs[2].plot(state_df["theta"], state_df["theta_dot"], label="State-Space Trajectory", color='green')
    axs[2].set_xlabel("θ (radians)")
    axs[2].set_ylabel("θ̇ (rad/s)")
    axs[2].set_title("State-Space Representation (θ vs. θ̇)")
    axs[2].legend()
    axs[2].grid()

    # Motor Position
    axs[3].plot(state_df["time"], state_df["x_pivot"], label="Cart Position (x_pivot)", color='purple')
    axs[3].set_ylabel("x_pivot (m)")
    axs[3].set_title("Motor Position Over Time")
    axs[3].legend()
    axs[3].grid()

    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # Run in manual mode by default
    simulate_pendulum(automated=False)
    
    # After simulation, plot the results
    if os.path.exists(motor_data_path):
        plot_motor_dynamics(motor_data_path)
    else:
        print(f"⚠️ Motor data not found at: {motor_data_path}")

    plot_system_state("reports/recording.csv")