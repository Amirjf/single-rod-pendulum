import time
import pygame
from Digital_twin import DigitalTwin

# Before starting run pip install -r requirements.txt

digital_twin = DigitalTwin()
        
if __name__=='__main__':
        running = True
        while running:
            
            theta, theta_dot, x_pivot = digital_twin.step()
            digital_twin.render(theta, x_pivot)
            time.sleep(digital_twin.delta_t)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in digital_twin.actions:
                        direction, duration = digital_twin.actions[event.key]
                        digital_twin.perform_action(direction, duration)

        pygame.quit()
