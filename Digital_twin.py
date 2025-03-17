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
        self.start_time = 0  # Store when the simulation starts

        # Initialize serial communication parameters
        self.ser = None
        self.device_connected = False

        # State configuration parameters
        self.steps = 0
        self.theta = 0     #np.pi-0.01
        self.theta_dot = 0.
        self.theta_double_dot = 0.
        self.x_pivot = 0
        self.delta_t = 0.005  # Example value, adjust as needed in seconds
        # self.delta_t = 0.0284  # Match sensor rate
        
        # Model parameters
        self.g = 9.8065     # Acceleration due to gravity (m/s^2)
        self.l = 0.3        # Length of the pendulum (m)
        self.c_air = 0.005    # Air friction coefficient
        self.c_c = 0.05      # Coulomb friction coefficient
        self.a_m = 0.5     # Motor acceleration force tranfer coefficient
        self.mc = 0.0       # Mass of the cart (kg)
        self.mp = 1       # Mass of the pendulum (kg)
        self.I = 0.00       # Moment of inertia of the pendulum (kg·m²)
        self.future_motor_accelerations = []
        self.future_motor_positions = []
        self.future_motor_velocities = []
        self.currentmotor_acceleration = 0.
        self.currentmotor_velocity = 0.
        self.time = 0.
        self.R_pulley = 0.05
        
        # Sensor data
        self.sensor_theta = 0
        self.current_sensor_motor_position = 0.
        self.current_action = 0
        
        # Keyboard action mappings
        _action_durations = [200, 150, 100, 50]  # Durations in milliseconds
        _keys_left = [pygame.K_a, pygame.K_s, pygame.K_d, pygame.K_f]
        _keys_right = [pygame.K_SEMICOLON, pygame.K_l, pygame.K_k , pygame.K_j]
        self.actions = {key: ('left', duration) for key, duration in zip(_keys_left, _action_durations)}
        self.actions.update({key: ('right', duration) for key, duration in zip(_keys_right, _action_durations)})
        self.action_map = [
            ('left', 0),  # No action
            ('left', 50), ('left', 100), ('left', 150), ('left', 200),
            ('right', 50), ('right', 100), ('right', 150), ('right', 200)
        ]
        self.recording = False
        self.writer = None
        self.start_time = 0.
        self.df = None
        
        # Initialize a pygame window
        self.initialize_pygame_window()

    def initialize_pygame_window(self):
        # Set up the drawing window
        pygame.init()
        self.screen = pygame.display.set_mode([1000, 800])

    def connect_device(self, port='COM3', baudrate=115200):
        # Establish a serial connection for sensor data
        self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=0, writeTimeout=0)
        self.device_connected = True
        print("Connected to: " + self.ser.portstr)

    def read_data(self):
        line = self.ser.readline()
        line = line.decode("utf-8")
        try:
            if len(line) > 2 and line != '-':
                sensor_data = line.split(",")
                if len(sensor_data[0]) > 0 and len(sensor_data[3]) > 0:
                    self.sensor_theta = int(sensor_data[0])
                    self.current_sensor_motor_position = -int(sensor_data[3])
        except Exception as e:
            print(e)
        if self.recording:
            self.writer.writerow([round(time.time() * 1000)-self.start_time, self.sensor_theta, self.current_sensor_motor_position])

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
        self.file = open('{}.csv'.format(name), 'w', newline='')  
        self.writer = csv.writer(self.file)
        self.start_time = round(time.time() * 1000)
        self.writer.writerow(["time", "theta", "x_pivot"])

    def stop_recording(self):
        self.recording = False
        self.file.close()
    
    def load_recording(self, name):
        self.df = pd.read_csv('{}.csv'.format(name))
        print("recording is loaded")
    
    def recorded_step(self,i):
        a = self.df["time"].pop(i)
        b = self.df["theta"].pop(i)
        c = self.df["x_pivot"].pop(i)  
        return a, b, c

    def perform_action(self, direction, duration):
        # Send the command to the device.
        if self.device_connected:
            if direction == 'left':
                d = -duration
            else:
                d = duration
            self.ser.write(str(d).encode())
        if duration > 0:
            self.update_motor_accelerations_real(direction, duration/1000)

    def update_motor_accelerations_real(self, direction, duration):
        """
        Compute motor acceleration using real physics (motor torque equation),
        and ensure proper deceleration using active braking.
        """

        # Convert direction to numerical value
        direction = -1 if direction == 'left' else 1

        # Motor parameters
        k = 0.0174  # Motor torque constant (N·m/A)
        J = 8.5075e-6  # Moment of inertia (kg·m²)
        R = 8.18  # Motor resistance (Ω)
        V_i = 12.0  # Input voltage (V)

        # Define motion phases
        t1 = duration / 4  # Acceleration phase
        t2_d = duration / 4  # Deceleration phase
        t2 = duration - t2_d  # Start of deceleration
        tf = duration  # Total movement time
        time_values = np.arange(0.0, tf + self.delta_t, self.delta_t)

        omega_m = [0.0]  # Initial angular velocity

        # Compute acceleration, velocity, and position for each time step
        for t in time_values:
            omega = omega_m[-1]  # Use last computed velocity

            # Compute acceleration based on phase
            if t < t1:  # Acceleration phase
                a_m_1 =  (k * (V_i - k * omega)) / (J * R)
                alpha_m = -4*direction*a_m_1/(t1*t1) * t * (t-t1)
            elif t1 <= t < t2:  # Constant velocity phase
                alpha_m = 0.0
            else:  # Deceleration phase (Active braking)
                a_m_2 = (k * (V_i + k * omega)) / (J * R)  # Apply braking force
                alpha_m  = 4*direction*a_m_2/(t2_d*t2_d) * (t-t2) * (t-duration)
                

            self.future_motor_accelerations.append(alpha_m)

            # Update omega using Euler integration
            omega_next = omega + alpha_m * self.delta_t
            omega_m.append(omega_next)

        # Store values
        self.future_motor_velocities = list(cumulative_trapezoid(self.future_motor_accelerations, dx=self.delta_t, initial=0))
        self.future_motor_positions = list(cumulative_trapezoid(self.future_motor_velocities, dx=self.delta_t, initial=0))

        # Save acceleration, velocity, and position to CSV
        with open("motor_data.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["time_s", "alpha_m_rad_s2", "omega_m_rad_s", "theta_m_rad"])
            for i in range(len(time_values) - 2):  # Avoid index issues
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
            for i in range(len(time_values) - 2):  # Avoid index issues
                writer.writerow([time_values[i], self.future_motor_accelerations[i], _velocity[i], self.future_motor_positions[i]])

    print("Motor data saved to motor_data.csv")
        
    def get_theta_double_dot(self, theta, theta_dot):
        """
        Lab 1: Model the angular acceleration (theta_double_dot) 
        as a function of theta, theta_dot and the self.currentmotor_acceleration. 
        You should include the following constants as well: c_air, c_c, a_m, l and g. 
        """
        torque_gravity = -(self.mp * self.g * self.l / (self.I + self.mp * self.l**2)) * np.sin(theta)
        torque_air_friction = -(self.c_air / (self.I + self.mp * self.l**2)) * theta_dot
        torque_coulomb_friction = -(self.c_c / (self.I + self.mp * self.l**2)) * theta_dot
        xdoubledot = self.a_m * self.R_pulley * self.currentmotor_acceleration
        torque_motor = - (self.mp * self.l/ (self.I + self.mp * self.l**2)) * xdoubledot * np.cos(theta)        
        angular_acceleration = torque_gravity + torque_air_friction + torque_coulomb_friction + torque_motor
    
        return angular_acceleration

    def step(self):
        # Get the predicted motor acceleration for the next step and the shift in x_pivot
        self.check_prediction_lists()
        #print(self.future_motor_accelerations)
        self.currentmotor_acceleration = self.future_motor_accelerations.pop(0)
        self.currentmotor_velocity = self.future_motor_velocities.pop(0)
        print("old x pivot:", self.x_pivot)
        self.x_pivot = self.x_pivot + self.R_pulley * self.future_motor_positions.pop(0)
        print("new x pivot:", self.x_pivot)
        # Update the system state based on the action and model dynamics
        self.theta_double_dot = self.get_theta_double_dot(self.theta, self.theta_dot)
        self.theta += self.theta_dot * self.delta_t
        self.theta_dot += self.theta_double_dot * self.delta_t
        self.time += self.delta_t
        self.steps += 1
        return self.theta, self.theta_dot, self.x_pivot, self.currentmotor_acceleration
        
    def draw_line_and_circles(self, colour, start_pos, end_pos, line_width=5, circle_radius=9):
        # Draw the line
        pygame.draw.line(self.screen, colour, start_pos, end_pos, line_width)
        
        # Draw square for cart (pivot point)
        square_size = 20  # Increased size of the square
        pygame.draw.rect(self.screen, colour, 
                        (start_pos[0] - square_size//2, 
                         start_pos[1] - square_size//2,
                         square_size, square_size), 2)  # Added width=2 for outline only
        
        # Draw circle for pendulum joint
        pygame.draw.circle(self.screen, colour, end_pos, circle_radius)

    def draw_pendulum(self, colour ,x, y, x_pivot):
        self.draw_line_and_circles(colour, [x_pivot+500, 400], [y+x_pivot+500, x+400])
        
    def draw_info_panel(self, surface, elapsed_time, theta, theta_dot, x_pivot, motor_acceleration, last_action):
        """Draw semi-transparent info panel with system information"""
        # Create semi-transparent surface with increased width
        info_surface = pygame.Surface((400, 150), pygame.SRCALPHA)
        pygame.draw.rect(info_surface, (0, 0, 0, 128), (0, 0, 400, 150))
        
        # Draw data with modern font
        font = pygame.font.Font(None, 24)
        text_color = (255, 255, 255)  # White text for better contrast
        
        text_lines = [
            f"Time Elapsed: {elapsed_time:.2f} s",
            f"Pendulum Angle: {np.degrees(theta):.1f}°",
            f"Angular Velocity: {theta_dot:.2f} rad/s",
            f"Cart Position: {x_pivot:.2f} cm",
            f"Motor Acceleration: {motor_acceleration:.2f} m/s²",
            f"Last Action: {last_action}"
        ]
        
        for i, line in enumerate(text_lines):
            text_surface = font.render(line, True, text_color)
            info_surface.blit(text_surface, (10, 10 + i * 20))
        
        surface.blit(info_surface, (20, 10))

    def render(self, theta, x_pivot, last_action="None"):
        """
        Render the pendulum system with enhanced visualization features.
        """
        # Ensure self.start_time is set when simulation begins
        if self.start_time == 0:
            self.start_time = time.time()

        # Clear the screen (white background)
        self.screen.fill((255, 255, 255))

        # Draw grid first (as background)
        self.draw_grid(self.screen)

        # Draw pendulum
        l = 100  # Length of the pendulum
        self.draw_pendulum((0, 0, 0), math.cos(theta) * l, math.sin(theta) * l, x_pivot)

        # Draw black line and circles for horizontal axis
        pygame.draw.line(self.screen, (0, 0, 0), [400, 400], [600, 400], 5)
        # Draw filled circles at both ends
        circle_radius = 9
        pygame.draw.circle(self.screen, (0, 0, 0), (400, 400), circle_radius)
        pygame.draw.circle(self.screen, (0, 0, 0), (600, 400), circle_radius)

        # Draw indicators (gravity arrow, cart center, auto-stabilization)
        self.draw_indicators(self.screen, theta, x_pivot, False)  # Replace False with actual auto-stabilization state

        # Draw semi-transparent info panel
        elapsed_time = time.time() - self.start_time
        self.draw_info_panel(self.screen, elapsed_time, theta, self.theta_dot, x_pivot, 
                           self.currentmotor_acceleration, last_action)

        # Update the display
        pygame.display.flip()

    def check_prediction_lists(self):
        if len(self.future_motor_accelerations) == 0:
            self.future_motor_accelerations = [0]
        if len(self.future_motor_velocities) == 0:
            self.future_motor_velocities = [0]
        if len(self.future_motor_positions) == 0:
            self.future_motor_positions = [0]

    def draw_arrow(self, surface, color, start_pos, end_pos, width=5):
        """Draw an arrow from start_pos to end_pos"""
        # Draw the main line
        pygame.draw.line(surface, color, start_pos, end_pos, width)
        
        # Calculate arrow head points
        angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
        arrow_length = 20
        arrow_width = 10
        
        # Calculate arrow head points
        arrow_point1 = (
            end_pos[0] - arrow_length * math.cos(angle - math.pi/6),
            end_pos[1] - arrow_length * math.sin(angle - math.pi/6)
        )
        arrow_point2 = (
            end_pos[0] - arrow_length * math.cos(angle + math.pi/6),
            end_pos[1] - arrow_length * math.sin(angle + math.pi/6)
        )
        
        # Draw arrow head
        pygame.draw.polygon(surface, color, [end_pos, arrow_point1, arrow_point2])

    def draw_indicators(self, surface, theta, x_pivot, auto_stabilization):
        # Draw cart center indicator as a square
        cart_center = (500 + x_pivot, 400)
        square_size = 6  # Size of the square
        pygame.draw.rect(surface, (0, 255, 0), 
                        (cart_center[0] - square_size//2, 
                         cart_center[1] - square_size//2,
                         square_size, square_size))

    def draw_grid(self, surface):
        # Draw vertical grid lines
        for x in range(0, 1000, 50):
            pygame.draw.line(surface, (200, 200, 200), (x, 0), (x, 800))
        
        # Draw horizontal grid lines
        for y in range(0, 800, 50):
            pygame.draw.line(surface, (200, 200, 200), (0, y), (1000, y))
        
        # Draw center reference line
        pygame.draw.line(surface, (255, 0, 0), (500, 0), (500, 800), 2)