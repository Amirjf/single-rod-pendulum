import time
import pygame
from Digital_twin import DigitalTwin
import csv

# Before starting run: pip install -r requirements.txt

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

# Plot the data
# Load the data from the CSV file
import pandas as pd
import matplotlib.pyplot as plt

# Load data
# Load data
data = pd.read_csv('recording.csv', header=None, names=['time', 'theta', 'theta_dot', 'x_pivot', 'acceleration'])

# Create figure and axis
fig, ax1 = plt.subplots(figsize=(10, 5))

# First y-axis for x_pivot
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('x_pivot', color='blue')
ax1.plot(data['time'], data['x_pivot'], label='x_pivot', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Second y-axis for theta_dot
ax2 = ax1.twinx()
ax2.set_ylabel('theta_dot', color='red')
ax2.plot(data['time'], data['theta_dot'], label='theta_dot', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Third y-axis for acceleration
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("outward", 60))  # Offset third y-axis
ax3.set_ylabel('Acceleration', color='green')
ax3.plot(data['time'], data['acceleration'], label='Acceleration', color='green')
ax3.tick_params(axis='y', labelcolor='green')

# Title and Grid
plt.title('x_pivot, theta_dot, and Acceleration over Time')
ax1.grid(True)

# Legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax3.legend(loc='lower right')

# Show plot
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV
df = pd.read_csv("motor_data.csv")

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 8))

# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.5) 

# Acceleration Plot
axs[0].plot(df["time_s"], df["alpha_m_rad_s2"], label="Motor Acceleration (α_m)", color='blue')
axs[0].set_ylabel("α_m (rad/s²)")
axs[0].set_title("Motor Acceleration Over Time")
axs[0].legend()
axs[0].grid()

# Velocity Plot
axs[1].plot(df["time_s"], df["omega_m_rad_s"], label="Motor Velocity (ω_m)", color='red')
axs[1].set_ylabel("ω_m (rad/s)")
axs[1].set_title("Motor Velocity Over Time") 
axs[1].legend()
axs[1].grid()

# Position Plot
axs[2].plot(df["time_s"], df["theta_m_rad"], label="Motor Angle (θ_m)", color='green')
axs[2].set_ylabel("θ_m (radians)")
axs[2].set_xlabel("Time (s)")
axs[2].set_title("Motor Angle Displacement Over Time")
axs[2].legend()
axs[2].grid()

# Overall Title
plt.suptitle("Motor Acceleration, Velocity, and Angle")

# Show the plot
plt.show()

# Load the data from CSV
state_df = pd.read_csv("recording.csv", header=None, names=["time", "theta", "theta_dot", "x_pivot", "acceleration"])

# Debugging: Print min/max values to check x_pivot
print("x_pivot Min:", state_df["x_pivot"].min(), "Max:", state_df["x_pivot"].max())

# Create subplots for State-Space Visualization
fig, axs = plt.subplots(4, 1, figsize=(10, 8), constrained_layout=True)

# Time-domain plot for Angle (theta)
axs[0].plot(state_df["time"], state_df["theta"], label="Angle (θ)", color='blue')
axs[0].set_ylabel("θ (radians)")
axs[0].set_title("Pendulum Angle Over Time")
axs[0].legend()
axs[0].grid()

# Time-domain plot for Angular Velocity (theta_dot)
axs[1].plot(state_df["time"], state_df["theta_dot"], label="Angular Velocity (θ̇)", color='red')
axs[1].set_ylabel("θ̇ (rad/s)")
axs[1].set_title("Pendulum Angular Velocity Over Time") 
axs[1].legend()
axs[1].grid()

# Phase Portrait: Theta vs Theta_dot
axs[2].plot(state_df["theta"], state_df["theta_dot"], label="State-Space Trajectory", color='green')
axs[2].set_xlabel("θ (radians)")
axs[2].set_ylabel("θ̇ (rad/s)")
axs[2].set_title("State-Space Representation (θ vs. θ̇)")
axs[2].legend()
axs[2].grid()

# Plot x_pivot over time
axs[3].plot(state_df["time"], state_df["x_pivot"], label="x_pivot", color='purple')
axs[3].set_ylabel("x_pivot (m)")
axs[3].set_title("Motor Position Over Time")
axs[3].legend()
axs[3].grid()

# Adjust layout and spacing
plt.tight_layout()

# Save figure
plt.savefig("state_space_analysis.png")
plt.show()
