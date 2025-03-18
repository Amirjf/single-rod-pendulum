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
        # Initialize Pygame parameters
        self.screen = None
        pygame.init()
        self.screen = pygame.display.set_mode([1000, 600])
        self.start_time = 0

        # Communication
        self.ser = None
        self.device_connected = False

        # Physics state
        self.steps = 0
        self.theta = 0                                                        # Pendulum angle in radians
        self.theta_dot = 0.                                                   # Angular velocity
        self.theta_double_dot = 0.                                           # Angular acceleration
        self.x_pivot = 0                                                     # Cart position
        self.delta_t = 0.005                                                 # Time step in seconds
        # self.delta_t = 0.0284                                              # Match sensor rate
        
        # Physical parameters
        self.g = 9.8065                                                      # Gravity (m/s^2)
        self.l = 0.3                                                         # Pendulum length (m)
        self.c_air = 0.005                                                   # Air friction coef
        self.c_c = 0.05                                                      # Coulomb friction coef
        self.a_m = 0.5                                                       # Motor force transfer coef
        self.mc = 0.0                                                        # Cart mass (kg)
        self.mp = 1                                                          # Pendulum mass (kg)
        self.I = 0.00                                                        # Moment of inertia (kg·m²)
        self.R_pulley = 0.05                                                 # Pulley radius (m)
        
        # Motor state
        self.future_motor_accelerations = []
        self.future_motor_positions = []
        self.future_motor_velocities = []
        self.currentmotor_acceleration = 0.
        self.currentmotor_velocity = 0.
        self.time = 0.
        
        # Sensor data
        self.sensor_theta = 0                                                # Sensor angle reading
        self.current_sensor_motor_position = 0.                              # Current motor position
        self.current_action = 0
        self.click_counter = 0                                              # Counter for number of clicks/events

        # Action configuration
        _action_durations = [200, 150, 100, 50]                             # Action durations (ms)
        _keys_left = [pygame.K_a, pygame.K_s, pygame.K_d, pygame.K_f]       # Left movement keys
        _keys_right = [pygame.K_SEMICOLON, pygame.K_l, pygame.K_k, pygame.K_j]  # Right movement keys
        
        self.actions = {key: ('left', duration) for key, duration in zip(_keys_left, _action_durations)}
        self.actions.update({key: ('right', duration) for key, duration in zip(_keys_right, _action_durations)})
        self.action_map = [
            ('left', 0),
            ('left', 50), ('left', 100), ('left', 150), ('left', 200),
            ('right', 50), ('right', 100), ('right', 150), ('right', 200)
        ]

        # Recording
        self.recording = False
        self.writer = None

        self.df = None

    def connect_device(self, port='COM3', baudrate=115200):                  # Setup serial connection
        self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=0, writeTimeout=0)
        self.device_connected = True
        print("Connected to: " + self.ser.portstr)

    def read_data(self):                                                     # Read sensor data
        line = self.ser.readline().decode("utf-8")
        try:
            if len(line) > 2 and line != '-':
                sensor_data = line.split(",")
                if len(sensor_data[0]) > 0 and len(sensor_data[3]) > 0:
                    self.sensor_theta = int(sensor_data[0])
                    self.current_sensor_motor_position = -int(sensor_data[3])
        except Exception as e:
            print(e)
        if self.recording:
            self.writer.writerow([round(time.time() * 1000)-self.start_time, 
                                self.sensor_theta, self.current_sensor_motor_position])

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
        
    def start_recording(self, name):
        # If you are working on the bonus assignments then you should also add a columb for actions (and safe those).
        self.recording = True
        self.file = open(f'{name}.csv', 'w', newline='')  
        self.writer = csv.writer(self.file)
        self.start_time = round(time.time() * 1000)
        self.writer.writerow(["time", "theta", "x_pivot"])

    def stop_recording(self):                                                # Stop data recording
        self.recording = False
        self.file.close()
    
    def load_recording(self, name):                                          # Load recorded data
        self.df = pd.read_csv(f'{name}.csv')
        print("recording is loaded")
    
    def recorded_step(self, i):                                             # Get recorded timestep
        return (self.df["time"].pop(i), 
                self.df["theta"].pop(i), 
                self.df["x_pivot"].pop(i))

    def perform_action(self, direction, duration):                           # Execute motor command
        if self.device_connected:
            d = -duration if direction == 'left' else duration
            self.ser.write(str(d).encode())
        if duration > 0:
            self.update_motor_accelerations_real(direction, duration/1000)
            self.click_counter += 1                                          # Increment click counter

    def update_motor_accelerations_real(self, direction, duration):          # Real motor dynamics
        direction = -1 if direction == 'left' else 1
        k = 0.0174                                                          # Motor torque constant (N·m/A)
        J = 8.5075e-6                                                       # Moment of inertia (kg·m²)
        R = 8.18                                                            # Motor resistance (Ω)
        V_i = 12.0                                                          # Input voltage (V)

        t1 = duration / 4                                                   # Acceleration phase
        t2_d = duration / 4                                                 # Deceleration phase
        t2 = duration - t2_d                                               # Start of deceleration
        tf = duration                                                       # Total movement time
        time_values = np.arange(0.0, tf + self.delta_t, self.delta_t)
        omega_m = [0.0]                                                     # Initial motor velocity

        # Calculate motor acceleration
        for t in time_values:
            omega = omega_m[-1]
            if t < t1:                                                      # Accelerate
                a_m_1 = (k * (V_i - k * omega)) / (J * R)
                alpha_m = -4*direction*a_m_1/(t1*t1) * t * (t-t1)
            elif t1 <= t < t2:                                             # Coast
                alpha_m = 0.0
            else:                                                          # Brake
                a_m_2 = (k * (V_i + k * omega)) / (J * R)
                alpha_m = 4*direction*a_m_2/(t2_d*t2_d) * (t-t2) * (t-duration)

            self.future_motor_accelerations.append(alpha_m)
            omega_m.append(omega + alpha_m * self.delta_t)                 # Euler integration

        # Calculate velocities and positions
        self.future_motor_velocities = list(cumulative_trapezoid(
            self.future_motor_accelerations, dx=self.delta_t, initial=0))
        self.future_motor_positions = list(cumulative_trapezoid(
            self.future_motor_velocities, dx=self.delta_t, initial=0))

        # Save data
        with open("motor_data.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["time_s", "alpha_m_rad_s2", "omega_m_rad_s", "theta_m_rad"])
            for i in range(len(time_values) - 2):
                writer.writerow([time_values[i], self.future_motor_accelerations[i], self.future_motor_velocities[i], self.future_motor_positions[i]])


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
        with open("motor_data.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["time_s", "alpha_m_rad_s2", "omega_m_rad_s", "theta_m_rad"])
            for i in range(len(time_values) - 2):
                writer.writerow([time_values[i], self.future_motor_accelerations[i], _velocity[i], self.future_motor_positions[i]])

    print("Motor data saved to motor_data.csv")
        
    def get_theta_double_dot(self, theta, theta_dot):
    
        torque_gravity = -(self.mp * self.g * self.l / (self.I + self.mp * self.l**2)) * np.sin(theta)
        torque_air_friction = -(self.c_air / (self.I + self.mp * self.l**2)) * theta_dot
        torque_coulomb_friction = -(self.c_c / (self.I + self.mp * self.l**2)) * theta_dot
        xdoubledot = self.a_m * self.R_pulley * self.currentmotor_acceleration
        torque_motor = -(self.mp * self.l/ (self.I + self.mp * self.l**2)) * xdoubledot * np.cos(theta)        
        
        return torque_gravity + torque_air_friction + torque_coulomb_friction + torque_motor

    def step(self):                                                          # Update simulation
        # Update motor state
        self.check_prediction_lists()
        self.currentmotor_acceleration = self.future_motor_accelerations.pop(0)
        self.currentmotor_velocity = self.future_motor_velocities.pop(0)
        self.x_pivot += self.R_pulley * self.future_motor_positions.pop(0)

        # Update pendulum state
        self.theta_double_dot = self.get_theta_double_dot(self.theta, self.theta_dot)
        self.theta += self.theta_dot * self.delta_t
        self.theta_dot += self.theta_double_dot * self.delta_t
        self.time += self.delta_t
        self.steps += 1
        
        return self.theta, self.theta_dot, self.x_pivot, self.currentmotor_acceleration
        
    def draw_line_and_circles(self, colour, start_pos, end_pos, line_width=5, circle_radius=9):
        pygame.draw.line(self.screen, colour, start_pos, end_pos, line_width)
        
        # Draw cart
        square_size = 20
        pygame.draw.rect(self.screen, colour,                               # Unfilled square
                        (start_pos[0] - square_size//2, 
                         start_pos[1] - square_size//2,
                         square_size, square_size), 2)
        
        pygame.draw.circle(self.screen, colour, end_pos, circle_radius)     # Joint

    def draw_pendulum(self, colour, x, y, x_pivot):                         # Draw pendulum system
        self.draw_line_and_circles(colour, [x_pivot+500, 400], [y+x_pivot+500, x+400])
        
    def draw_info_panel(self, surface, elapsed_time, theta, theta_dot, x_pivot, motor_acceleration, last_action):
        # Create panel
        info_surface = pygame.Surface((400, 150), pygame.SRCALPHA)
        pygame.draw.rect(info_surface, (0, 0, 0, 128), (0, 0, 400, 150))
        
        # Draw text
        font = pygame.font.Font(None, 24)
        
        # Normalize angle to 0-360 degrees
        angle_degrees = np.degrees(theta)
        normalized_angle = angle_degrees % 360
        
        text_lines = [
            f"Time Elapsed: {elapsed_time:.2f} s",
            f"Pendulum Angle: {normalized_angle:.1f}°",
            f"Angular Velocity: {theta_dot:.2f} rad/s",
            f"Cart Position: {x_pivot:.2f} cm",
            f"Motor Acceleration: {motor_acceleration:.2f} m/s²",
            f"Last Action: {last_action}",
            f"Number of Clicks: {self.click_counter}"
        ]
        
        for i, line in enumerate(text_lines):
            text_surface = font.render(line, True, (255, 255, 255))
            info_surface.blit(text_surface, (10, 10 + i * 20))
        
        surface.blit(info_surface, (20, 10))

    def draw_key_actions(self, surface, last_action):                        # Draw key actions panel
        # Create semi-transparent panel
        panel_width = 100
        panel_height = 250
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        pygame.draw.rect(panel_surface, (0, 0, 0, 128), (0, 0, panel_width, panel_height))
        
        # Draw title
        font = pygame.font.Font(None, 20)  # Smaller font
        title = font.render("Key Actions", True, (255, 255, 255))
        panel_surface.blit(title, (10, 5))
        
        # Draw left actions
        y_pos = 30
        left_text = font.render("Left:", True, (255, 255, 255))
        panel_surface.blit(left_text, (10, y_pos))
        y_pos += 20
        
        for key, (direction, duration) in self.actions.items():
            if direction == 'left':
                key_name = pygame.key.name(key).upper()
                # Compare with last_action directly
                action_str = f"{direction.capitalize()} ({duration} ms)"
                color = (0, 255, 0) if last_action == action_str else (255, 255, 255)
                
                text = f"{key_name}: {duration}ms"
                text_surface = font.render(text, True, color)
                panel_surface.blit(text_surface, (20, y_pos))
                y_pos += 20
        
        # Draw right actions
        y_pos += 10  # Add some space between left and right sections
        right_text = font.render("Right:", True, (255, 255, 255))
        panel_surface.blit(right_text, (10, y_pos))
        y_pos += 20
        
        for key, (direction, duration) in self.actions.items():
            if direction == 'right':
                key_name = pygame.key.name(key).upper()
                # Compare with last_action directly
                action_str = f"{direction.capitalize()} ({duration} ms)"
                color = (0, 255, 0) if last_action == action_str else (255, 255, 255)
                
                text = f"{key_name}: {duration}ms"
                text_surface = font.render(text, True, color)
                panel_surface.blit(text_surface, (20, y_pos))
                y_pos += 20
        
        # Position panel in top right corner
        surface.blit(panel_surface, (surface.get_width() - panel_width - 10, 10))

    def render(self, theta, x_pivot, last_action="None"):                   # Main render loop
        if self.start_time == 0:
            self.start_time = time.time()

        # Clear and draw background
        self.screen.fill((255, 255, 255))
        self.draw_grid(self.screen)

        # Draw pendulum system
        l = 100                                                             # Pendulum length (pixels)
        self.draw_pendulum((0, 0, 0), math.cos(theta) * l, math.sin(theta) * l, x_pivot)

        # Draw track
        pygame.draw.line(self.screen, (0, 0, 0), [400, 400], [600, 400], 5)
        pygame.draw.circle(self.screen, (0, 0, 0), (400, 400), 9)          # End stops
        pygame.draw.circle(self.screen, (0, 0, 0), (600, 400), 9)

        # Draw UI elements
        self.draw_indicators(self.screen, theta, x_pivot, False)
        elapsed_time = time.time() - self.start_time
        self.draw_info_panel(self.screen, elapsed_time, theta, self.theta_dot, x_pivot, 
                           self.currentmotor_acceleration, last_action)
        
        # Draw key actions panel
        self.draw_key_actions(self.screen, last_action)

        pygame.display.flip()

    def check_prediction_lists(self):                                       # Initialize empty lists
        if len(self.future_motor_accelerations) == 0:
            self.future_motor_accelerations = [0]
        if len(self.future_motor_velocities) == 0:
            self.future_motor_velocities = [0]
        if len(self.future_motor_positions) == 0:
            self.future_motor_positions = [0]

    def draw_indicators(self, surface, theta, x_pivot, auto_stabilization): # Draw cart indicator
        cart_center = (500 + x_pivot, 400)
        square_size = 6
        pygame.draw.rect(surface, (0, 255, 0),
                        (cart_center[0] - square_size//2, 
                         cart_center[1] - square_size//2,
                         square_size, square_size))

    def draw_grid(self, surface):                                          # Draw background grid
        for x in range(0, 1000, 50):                                       # Vertical lines
            pygame.draw.line(surface, (200, 200, 200), (x, 0), (x, 800))
        for y in range(0, 800, 50):                                        # Horizontal lines
            pygame.draw.line(surface, (200, 200, 200), (0, y), (1000, y))
        pygame.draw.line(surface, (255, 0, 0), (500, 0), (500, 800), 2)   # Center line