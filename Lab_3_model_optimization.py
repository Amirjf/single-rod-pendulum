import numpy as np
from Digital_twin import DigitalTwin
import pandas as pd

digital_twin = DigitalTwin()
digital_twin.get_theta_double_dot()
# Path to the CSV file
csv_file_path = 'SOME_data.csv'
df = pd.read_csv(csv_file_path)
df_time = df['time'] 
df_theta = df['theta']

#Process dt_theta sush that it is translated to radians.

def find_initial_state(df_theta, df_time):
     # Find the initial condions of theta and theta_dot in the data
     theta = 0
     theta_dot = 0
     return theta, theta_dot

#Initial conditions based on the recorded data
theta, theta_dot = find_initial_state(df_theta, df_time)
#should be the same as your recording, sim_time is equal to total time of the recording
delta_t = 0.01
sim_time = 0.

# Define the ranges for each parameter and the 'resolution' of the search space
c_air_range = np.linspace(0.0, 0.8, 10)  # Example range for c_air
c_c_range = np.linspace(0.6, 1.2, 10)    # Example range for c_c
g = 9.81  # Assuming g is constant
l_range = np.linspace(0.1, 0.5, 10)      # Example range for l

def simulate_potential_model(theta, theta_dot, c_air, c_c, g, l, theta_measurements):
     digital_twin.c_air =c_air
     digital_twin.c_c = c_c
     digital_twin.g = g
     digital_twin.l = l
     sim_measurements = []
     for i in range(len(theta_measurements)):
        theta_double_dot = digital_twin.get_theta_double_dot(theta, theta_dot)
        theta += theta_dot * delta_t
        theta_dot += theta_double_dot * delta_t
        sim_measurements.append(theta)
     rmse = np.sqrt(np.mean((np.array(theta_measurements) - np.array(sim_measurements))**2))
     return rmse

#example usage
exp_err = simulate_potential_model(np.pi-0.5, -0.1, 0.16, 1.1, 9.81 ,0.3, df_theta)
# Initialize variables to store the best parameters and the lowest error found
best_params = None
lowest_error = float('inf')

# Nested loops to go through each combination of parameters
for c_air in c_air_range:
        for c_c in c_c_range:
            for l in l_range:
                error = simulate_potential_model(theta, theta_dot, c_air, c_c, g, l, df_theta)
                if error < lowest_error:
                    lowest_error = error
                    print(error, "found a better error")
                    best_params = (c_air, c_c, l)

print("Best Parameters:", best_params)
print("Lowest Error:", lowest_error)
print("expected Error: ", exp_err)
