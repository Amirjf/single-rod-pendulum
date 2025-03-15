import time
import pygame
from Digital_twin import DigitalTwin
import csv

# Before starting run pip install -r requirements.txt

# Clear the contents of the recording.csv file
with open('recording.csv', mode='w', newline='') as file:
    file.truncate()

digital_twin = DigitalTwin()
        
if __name__=='__main__':
        running = True
        while running:
            
            theta, theta_dot, x_pivot = digital_twin.step()
            digital_twin.render(theta, x_pivot)
            time.sleep(digital_twin.delta_t)

            # Save the theta, theta_dot, x_pivot to plot later
            with open('recording.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([time.time(), theta, theta_dot, x_pivot])
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in digital_twin.actions:
                        direction, duration = digital_twin.actions[event.key]
                        digital_twin.perform_action(direction, duration)
                    elif event.key == pygame.K_r:
                            digital_twin = DigitalTwin()  # Restart the system
                            print("System restarted")
                    elif event.key == pygame.K_ESCAPE:
                        running = False # Quit the simulation

        pygame.quit()

# Plot the data
# Load the data from the CSV file
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('recording.csv', header=None, names=['time', 'theta', 'theta_dot', 'x_pivot'])

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

# Title and Grid
plt.title('x_pivot and theta_dot over Time')
ax1.grid(True)

# Legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Show plot
plt.show()
