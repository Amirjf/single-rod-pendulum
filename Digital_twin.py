# Import the necessary libraries
import pygame
import serial
import numpy as np
import csv
import math
from scipy.integrate import cumulative_trapezoid
import time
import pandas as pd
import csv

class DigitalTwin:
    def __init__(self):
        # Pygame Display
        self.screen = None  # Pygame window for visualization

        # Serial Communication
        self.ser = None  # Serial connection object
        self.device_connected = False  # Connection status flag

        # Physics State
        self.steps = 0  # Simulation time step counter
        self.theta = 1.5   # Pendulum angle (radians)
        self.theta_dot = 0.  # Angular velocity (rad/s)
        self.theta_double_dot = 0.  # Angular acceleration (rad/s²)
        self.x_pivot = 0  # Cart position (m)
        self.delta_t = 0.005  # Time step (s) - optimized for visualization
        self.k = 0.0174  
        # self.delta_t = 0.0284  # Alternative time step for sensor matching

        # Model Parameters
        self.g = 9.8065  # Gravity (m/s²)
        self.l = 0.35  # Pendulum length (m)
        self.c_air = 0.18  # Air friction coefficient
        self.c_c = 0.0028  # Coulomb friction coefficient
        self.a_m = 0.5  # Motor force transfer coefficient
        self.mc = 0.0  # Cart mass (kg)
        self.mp = 1  # Pendulum mass (kg)
        self.I = 0.00  # Moment of inertia (kg·m²)
        self.R_pulley = 0.05  # Pulley radius (m)

        # Motor State
        self.future_motor_accelerations = []
        self.future_motor_positions = []
        self.future_motor_velocities = []
        self.currentmotor_acceleration = 0.
        self.currentmotor_velocity = 0.
        self.time = 0.

        # Sensor Data
        self.sensor_theta = 0  # Measured pendulum angle
        self.current_sensor_motor_position = 0.  # Measured motor position
        self.current_action = 0  # Current applied action
        self.click_counter = 0  # User input counter

        # Action Configuration (User Input)
        action_durations = [200, 150, 100, 50]  # Action durations (ms)
        keys_left = [pygame.K_a, pygame.K_s, pygame.K_d, pygame.K_f]  # Left movement keys
        keys_right = [pygame.K_SEMICOLON, pygame.K_l, pygame.K_k, pygame.K_j]  # Right movement keys

        # Map user input keys to movement actions
        self.actions = {key: ('left', dur) for key, dur in zip(keys_left, action_durations)}
        self.actions.update({key: ('right', dur) for key, dur in zip(keys_right, action_durations)})

        # Define possible actions with different durations
        self.action_map = [
            ('left', 0), ('left', 50), ('left', 100), ('left', 150), ('left', 200),
            ('right', 50), ('right', 100), ('right', 150), ('right', 200)
        ]

        # Data Recording
        self.recording = False  # Recording status flag
        self.writer = None  # File writer for saving data
        self.start_time = 0  # Recording start time
        self.df = None  # Data storage

        # Initialize Pygame Window
        self.initialize_pygame_window()

    def initialize_pygame_window(self):  # Initialize and set up the Pygame window
        pygame.init()  # Initialize Pygame
        self.screen = pygame.display.set_mode([1000, 600])  # Create window (1000x600 pixels)

    def connect_device(self, port='/dev/cu.usbserial-0001', baudrate=115200):  # Establish a serial connection for sensor data
            self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=0, writeTimeout=0)  
            self.device_connected = True  # Mark device as connected
  
            print("Connected to: " + self.ser.portstr)  # Print confirmation message

    def read_data(self):  # Read and process sensor data from the serial connection
            line = self.ser.readline().decode("utf-8")  # Read serial data and decode
            try:
                if len(line) > 2 and line != '-':  # Validate incoming data
                    sensor_data = line.split(",")  # Split CSV data
                    if len(sensor_data[0]) > 0 and len(sensor_data[3]) > 0:  
                        self.sensor_theta = int(sensor_data[0])  # Extract pendulum angle
                        self.current_sensor_motor_position = -int(sensor_data[3])  # Extract motor position
            except Exception as e:
                print(e)  # Handle decoding errors
            
            if self.recording:  # If recording is enabled, save data
                self.writer.writerow([
                    round(time.time() * 1000) - self.start_time,  # Timestamp (ms)
                    self.sensor_theta,  # Recorded pendulum angle
                    self.current_sensor_motor_position  # Recorded motor position
                ])

    def process_data(self):
        """
        Lab 2: Use the sensor data retured by the function read_data. 
        The sensor data needs to be represented in the virtual model.
        First the data should be scaled and calibrated,
        Secondly noise should be reduced trough a filtering method.
        Return the processed data such that it can be used in visualization and recording.
        Also, transform the current_sensor_motor_position to be acurate. 
        This means that the encoder value should be scaled to match the displacement in the virtual model.
        """
        self.sensor_theta = 0
        self.current_sensor_motor_position = 0
        
    def start_recording(self, name):  # Start recording data and save it to a CSV file
        self.recording = True  
        self.file = open(f'{name}.csv', 'w', newline='')  # Open CSV file for writing
        self.writer = csv.writer(self.file)  
        self.start_time = round(time.time() * 1000)  # Capture recording start time (ms)
        self.writer.writerow(["time", "theta", "x_pivot"])  # Write header row

    def stop_recording(self):  # Stop data recording and close the file
            self.recording = False  
            self.file.close()  

    def load_recording(self, name):  # Load recorded data from CSV into a dataframe
            self.df = pd.read_csv(f'{name}.csv')  
            print("Recording is loaded")  # Confirm data loading

    def recorded_step(self, i):  # Retrieve a recorded timestep from the dataframe
            return (self.df["time"].pop(i),  
                    self.df["theta"].pop(i),  
                    self.df["x_pivot"].pop(i))  

    def perform_action(self, direction, duration):  # Execute motor command based on user input
            if self.device_connected:  # Send command only if device is connected
                d = -duration if direction == 'left' else duration  
                self.ser.write(str(d).encode())  # Send duration as a string via serial connection
            
            if duration > 0:  
                self.update_motor_accelerations_real(direction, duration/1000)  # Update motor acceleration
                self.click_counter += 1  # Increment click counter

    def update_motor_accelerations_real(self, direction, duration):  
        # Determine motion direction (-1 for left, 1 for right)
        direction = -1 if direction == 'left' else 1

        # Motor parameters
        k = self.k      # Motor torque constant (N·m/A)
        J = 8.5075e-6     # Moment of inertia (kg·m²)
        R = 8.18          # Motor resistance (Ω)
        V_i = 12.0        # Input voltage (V)
        B_v = 1.5e-8      # Viscous damping coefficient (kg·m²/s)
        T_q = 0.0         # Load torque (assumed zero for simplification)

        # Motion timing (Three-phase movement)
        t1 = duration / 4  # Acceleration phase
        t2_d = duration / 4  # Deceleration phase
        t2 = duration - t2_d  # Start of deceleration phase
        tf = duration  # Total movement duration
        
        # Time discretization
        time_values = np.arange(0.0, tf + self.delta_t, self.delta_t)
        omega_m = [0.0]  # Initial motor velocity

        # Compute motor acceleration over time
        for t in time_values:
            omega = omega_m[-1]
            alpha_m = ((k * (V_i - k * omega)) / (J * R)) - ((B_v * omega) / J) - T_q / J  # Motor acceleration
            
            if t < t1:  
                # Acceleration phase (Quadratic function for smooth start)
                alpha_m = -4 * direction * alpha_m / (t1 * t1) * t * (t - t1)
            elif t1 <= t < t2:  
                # Constant velocity phase (No acceleration)
                alpha_m = 0.0
            else:  
                # Deceleration phase (Mirrors acceleration)
                alpha_m = 4 * direction * alpha_m / (t2_d * t2_d) * (t - t2) * (t - duration)

            self.future_motor_accelerations.append(alpha_m)
            omega_m.append(omega + alpha_m * self.delta_t)  # Euler integration

        # Compute motor velocities and positions using numerical integration
        self.future_motor_velocities = list(cumulative_trapezoid(
            self.future_motor_accelerations, dx=self.delta_t, initial=0))
        self.future_motor_positions = list(cumulative_trapezoid(
            self.future_motor_velocities, dx=self.delta_t, initial=0))

        # Save motor data to CSV file
        with open("reports/motor_data.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["time_s", "alpha_m_rad_s2", "omega_m_rad_s", "theta_m_rad"])
            for i in range(len(time_values) - 2):
                writer.writerow([
                    time_values[i], 
                    self.future_motor_accelerations[i], 
                    self.future_motor_velocities[i], 
                    self.future_motor_positions[i]
                ])


    def update_motor_accelerations(self, direction, duration):
        if direction == 'left':
            direction = -1
        else:
            direction = 1

        """
        Lab 1 & 3 bonus: Model the expected acceleration response of the motor.  
        """
        a_m_1 = 0.05
        a_m_2 = 0.05
        t1 = duration/4
        t2_d = duration/4
        t2 = duration - t2_d

        time_values = np.arange(0.0, duration + self.delta_t, self.delta_t)

        for t in np.arange(0.0, duration+self.delta_t, self.delta_t):
            if t <= t1:
                c = -4*direction*a_m_1/(t1*t1) * t * (t-t1)
            elif t < t2 and t > t1:
                c = 0 
            elif t >= t2:
                c = 4*direction*a_m_2/(t2_d*t2_d) * (t-t2) * (t-duration)
            
            self.future_motor_accelerations.append(c)
        
        _velocity = cumulative_trapezoid(self.future_motor_accelerations,dx=self.delta_t, initial=0)
        self.future_motor_positions = list(cumulative_trapezoid(_velocity,dx=self.delta_t,initial=0))

        # Save acceleration, velocity, and position to CSV
        with open("reports/motor_data.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["time_s", "alpha_m_rad_s2", "omega_m_rad_s", "theta_m_rad"])
            for i in range(len(time_values) - 2):
                writer.writerow([time_values[i], self.future_motor_accelerations[i], _velocity[i], self.future_motor_positions[i]])

    print("Motor data saved to motor_data.csv")
        
    def get_theta_double_dot(self, theta, theta_dot):

        # Torque due to gravity (restoring force)
        torque_gravity = -(self.mp * self.g * self.l / (self.I + self.mp * self.l**2)) * np.sin(theta)

        # # Torque due to air friction (proportional to velocity)
        torque_air_friction = -(self.c_air / (self.I + self.mp * self.l**2)) * theta_dot

        # # Torque due to Coulomb friction (constant but direction-dependent)
        torque_coulomb_friction = -(self.c_c / (self.I + self.mp * self.l**2)) * theta_dot

        # Cart acceleration effect (linear-to-angular translation)
        xdoubledot = self.a_m * self.R_pulley * self.currentmotor_acceleration

        # Torque exerted by the motor on the pendulum
        torque_motor = -(self.mp * self.l / (self.I + self.mp * self.l**2)) * xdoubledot * np.cos(theta)        
        
        # Sum all torques to compute the total angular acceleration
        return torque_gravity + torque_air_friction + torque_coulomb_friction + torque_motor

    def step(self):  # Update simulation state at each timestep
        # Update motor state
        self.check_prediction_lists()  # Ensure prediction lists are not empty
        self.currentmotor_acceleration = self.future_motor_accelerations.pop(0)  # Get next motor acceleration
        self.currentmotor_velocity = self.future_motor_velocities.pop(0)  # Get next motor velocity
        self.x_pivot += self.R_pulley * self.future_motor_positions.pop(0)  # Update cart position

        # Update pendulum state
        self.theta_double_dot = self.get_theta_double_dot(self.theta, self.theta_dot)  # Compute angular acceleration
        self.theta += self.theta_dot * self.delta_t  # Update angle using angular velocity
        self.theta_dot += self.theta_double_dot * self.delta_t  # Update angular velocity
        self.time += self.delta_t  # Increment simulation time
        self.steps += 1  # Increment step counter
        
        return self.theta, self.theta_dot, self.x_pivot, self.currentmotor_acceleration  # Return updated state
        
    def draw_line_and_circles(self, colour, start_pos, end_pos, line_width=5, circle_radius=9):  # Draw pendulum arm and joint
        pygame.draw.line(self.screen, colour, start_pos, end_pos, line_width)  # Draw pendulum rod
        
        # Draw cart (represented as a square)
        square_size = 20  
        pygame.draw.rect(self.screen, colour,  
                        (start_pos[0] - square_size // 2,  
                         start_pos[1] - square_size // 2,  
                         square_size, square_size), 2)  # Unfilled square
        
        pygame.draw.circle(self.screen, colour, end_pos, circle_radius)  # Draw joint (circle)

    def draw_pendulum(self, colour, x, y, x_pivot):  # Draw complete pendulum system
            self.draw_line_and_circles(colour, [x_pivot + 500, 400], [y + x_pivot + 500, x + 400])  
        
    def draw_info_panel(self, surface, elapsed_time, theta, theta_dot, x_pivot, motor_acceleration, last_action):  # Draw system info panel
        info_surface = pygame.Surface((400, 150), pygame.SRCALPHA)  # Create a semi-transparent panel
        pygame.draw.rect(info_surface, (0, 0, 0, 128), (0, 0, 400, 150))  # Panel background
        
        font = pygame.font.Font(None, 24)  # Font settings

        # Normalize angle to 0-360 degrees
        angle_degrees = np.degrees(theta)  
        normalized_angle = angle_degrees % 360  

        # Define text content
        text_lines = [
            f"Time Elapsed: {elapsed_time:.2f} s",
            f"Pendulum Angle: {normalized_angle:.1f}°",
            f"Angular Velocity: {theta_dot:.2f} rad/s",
            f"Cart Position: {x_pivot:.2f} cm",
            f"Motor Acceleration: {motor_acceleration:.2f} m/s²",
            f"Last Action: {last_action}",
            f"Number of Clicks: {self.click_counter}"
        ]

        # Render and display text
        for i, line in enumerate(text_lines):
            text_surface = font.render(line, True, (255, 255, 255))  # White text
            info_surface.blit(text_surface, (10, 10 + i * 20))  # Position text

        surface.blit(info_surface, (20, 10))  # Display panel at the top left

    def draw_key_actions(self, surface, last_action):  # Draw key actions panel (left/right controls)
            panel_width = 100  
            panel_height = 250  
            panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)  
            pygame.draw.rect(panel_surface, (0, 0, 0, 128), (0, 0, panel_width, panel_height))  # Background
            
            font = pygame.font.Font(None, 20)  # Smaller font for actions
            panel_surface.blit(font.render("Key Actions", True, (255, 255, 255)), (10, 5))  # Title

            # Draw left key actions
            y_pos = 30  
            panel_surface.blit(font.render("Left:", True, (255, 255, 255)), (10, y_pos))  
            y_pos += 20  

            for key, (direction, duration) in self.actions.items():
                if direction == 'left':
                    key_name = pygame.key.name(key).upper()  
                    action_str = f"{direction.capitalize()} ({duration} ms)"  
                    color = (0, 255, 0) if last_action == action_str else (255, 255, 255)  # Highlight last action
                    
                    text_surface = font.render(f"{key_name}: {duration}ms", True, color)  
                    panel_surface.blit(text_surface, (20, y_pos))  
                    y_pos += 20  

            # Draw right key actions
            y_pos += 10  # Space between left and right sections
            panel_surface.blit(font.render("Right:", True, (255, 255, 255)), (10, y_pos))  
            y_pos += 20  

            for key, (direction, duration) in self.actions.items():
                if direction == 'right':
                    key_name = pygame.key.name(key).upper()  
                    action_str = f"{direction.capitalize()} ({duration} ms)"  
                    color = (0, 255, 0) if last_action == action_str else (255, 255, 255)  
                    
                    text_surface = font.render(f"{key_name}: {duration}ms", True, color)  
                    panel_surface.blit(text_surface, (20, y_pos))  
                    y_pos += 20  

            surface.blit(panel_surface, (surface.get_width() - panel_width - 10, 10))  # Position panel at top right

    def render(self, theta, x_pivot, last_action="None"):  # Main render loop
        if self.start_time == 0:
            self.start_time = time.time()  # Initialize start time if not set

        self.screen.fill((255, 255, 255))  # Clear screen with white background
        self.draw_grid(self.screen)  # Draw grid on background

        # Draw pendulum system
        l = 100  # Pendulum length in pixels
        self.draw_pendulum((0, 0, 0), math.cos(theta) * l, math.sin(theta) * l, x_pivot)  

        # Draw track
        pygame.draw.line(self.screen, (0, 0, 0), [400, 400], [600, 400], 5)  # Draw base track
        pygame.draw.circle(self.screen, (0, 0, 0), (400, 400), 9)  # Left end stop
        pygame.draw.circle(self.screen, (0, 0, 0), (600, 400), 9)  # Right end stop

        # Draw UI elements
        self.draw_indicators(self.screen, theta, x_pivot, False)  # Display angle and position indicators
        elapsed_time = time.time() - self.start_time  # Calculate elapsed time
        self.draw_info_panel(self.screen, elapsed_time, theta, self.theta_dot, x_pivot,  
                           self.currentmotor_acceleration, last_action)  # Display info panel
        
        self.draw_key_actions(self.screen, last_action)  # Draw key actions panel

        pygame.display.flip()  # Refresh display

    def check_prediction_lists(self):  # Ensure motor state lists are initialized
        if len(self.future_motor_accelerations) == 0:
            self.future_motor_accelerations = [0]  # Default acceleration value
        if len(self.future_motor_velocities) == 0:
            self.future_motor_velocities = [0]  # Default velocity value
        if len(self.future_motor_positions) == 0:
            self.future_motor_positions = [0]  # Default position value

    def draw_indicators(self, surface, theta, x_pivot, auto_stabilization):  # Draw cart position indicator
            cart_center = (500 + x_pivot, 400)  # Compute cart position
            square_size = 6  # Indicator size
            pygame.draw.rect(surface, (0, 255, 0),  
                            (cart_center[0] - square_size // 2,  
                            cart_center[1] - square_size // 2,  
                            square_size, square_size))  # Green square indicator

    def draw_grid(self, surface):  # Draw background grid for visualization
            for x in range(0, 1000, 50):  # Vertical grid lines
                pygame.draw.line(surface, (200, 200, 200), (x, 0), (x, 800))  
            for y in range(0, 800, 50):  # Horizontal grid lines
                pygame.draw.line(surface, (200, 200, 200), (0, y), (1000, y))  
            pygame.draw.line(surface, (255, 0, 0), (500, 0), (500, 800), 2)  # Center reference line

    
    def simulate_passive(self, theta0, theta_dot0, time_array):
        self.theta = theta0
        self.theta_dot = theta_dot0

        theta_history = [self.theta]

        for i in range(1, len(time_array)):
            dt = time_array[i] - time_array[i - 1]
            theta_ddot = self.get_theta_double_dot(self.theta, self.theta_dot)
            self.theta_dot += theta_ddot * dt
            self.theta += self.theta_dot * dt
            theta_history.append(self.theta)
            if i % 100 == 0:
                 print(f"t={time_array[i]:.2f}s | θ={self.theta:.4f}, θ̇={self.theta_dot:.4f}, θ̈={theta_ddot:.4f}")

        return np.array(theta_history)