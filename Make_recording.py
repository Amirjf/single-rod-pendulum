import time
import pygame
from Digital_twin import DigitalTwin

digital_twin = DigitalTwin()
        
if __name__=='__main__':
        running = True
        # You can test a sequence of actions (find the action map in the digitalTwin).
        # Each action is performed after 200ms so that the actions do not overlap in time.
        # Can also use your keyboard to manually control the system.
        actions =  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        digital_twin.connect_device()

        digital_twin.start_recording("test_data")
        
        while running:

            if digital_twin.steps%40 == 0 and len(actions) > 0:

                action = actions.pop(0)
                direction, duration = digital_twin.action_map[action]

                digital_twin.perform_action(direction, duration)
                
            digital_twin.read_data()
            
            theta, theta_dot, x_pivot, motor_acceleration = digital_twin.step()

            
            digital_twin.render(theta, x_pivot)
            
            time.sleep(digital_twin.delta_t)
            
            # if len(actions) == 0 and digital_twin.recording and digital_twin.time > 5:
            #      digital_twin.stop_recording()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        pygame.quit()
