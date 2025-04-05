import time
import pygame
from Digital_twin import DigitalTwin
import csv
import pandas as pd
import matplotlib.pyplot as plt


# Clear the contents of the recording.csv file
with open('recording.csv', mode='w', newline='') as file:
    file.truncate()

digital_twin = DigitalTwin()

if __name__ == '__main__':
    running = True
    last_action = "None"  # Track the last action performed

    while running:
        # Step through simulation
        theta, theta_dot, x_pivot, currentmotor_acceleration = digital_twin.step()

        # Render with updated information (pass last_action)
        digital_twin.render(theta, x_pivot, last_action) # Update to include action tracking

        # Sleep for time step
        time.sleep(digital_twin.delta_t)

        # Save the theta, theta_dot, x_pivot, and acceleration to CSV
        with open('recording.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([time.time(), theta, theta_dot, x_pivot, currentmotor_acceleration])

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in digital_twin.actions:
                    direction, duration = digital_twin.actions[event.key]
                    digital_twin.perform_action(direction, duration)
                    last_action = f"{direction.capitalize()} ({duration} ms)"  # Track last action
                elif event.key == pygame.K_r:
                    digital_twin = DigitalTwin()  # Restart the system
                    last_action = "System Restarted"
                    print("System restarted")
                elif event.key == pygame.K_ESCAPE:
                    running = False  # Quit the simulation

    pygame.quit()


def plot_motor_dynamics(csv_file, save_path="motor_Acc_vel_angle.png"):
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


def plot_system_state(csv_file, save_path="state_space_analysis.png"):
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

plot_motor_dynamics("motor_data.csv")
plot_system_state("recording.csv")