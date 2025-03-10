import time
import pygame
from Digital_twin import DigitalTwin

# Before starting run pip install -r requirements.txt

digital_twin = DigitalTwin()

# Set up font for displaying text
font = pygame.font.Font(None, 24)  # Default font, size 24

def draw_text(surface, text, position, color=(0, 0, 0)):
    """ Utility function to draw text on the screen. """
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, position)
        
if __name__=='__main__':
    running = True
    while running:
        theta, theta_dot, x_pivot = digital_twin.step()
        digital_twin.render(theta, x_pivot)
        # Get angular acceleration (theta_double_dot) from DigitalTwin class
        theta_double_dot = digital_twin.theta_double_dot  # This should be updated in `step()`
        currentmotor_acceleration = digital_twin.currentmotor_acceleration

        # Get the Pygame screen surface
        screen = pygame.display.get_surface()
        
        if screen:  # Make sure screen is initialized
            # Draw text for debugging information
            draw_text(screen, f"Theta: {theta:.2f} rad", (20, 20))
            draw_text(screen, f"Theta_dot: {theta_dot:.2f} rad/s", (20, 40))
            draw_text(screen, f"Theta_double_dot: {theta_double_dot:.2f} rad/s²", (20, 60))
            draw_text(screen, f"x_pivot: {x_pivot:.2f}", (20, 80))
            draw_text(screen, f"Motor Acceleration: {currentmotor_acceleration:.2f} m/s²", (20, 100))

        # Update Pygame display
        pygame.display.flip()

        time.sleep(digital_twin.delta_t)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in digital_twin.actions:
                    direction, duration = digital_twin.actions[event.key]
                    digital_twin.perform_action(direction, duration)
                # Add a key to toggle auto-stabilization (e.g., space bar)
                # elif event.key == pygame.K_SPACE:
                #     digital_twin.toggle_auto_stabilization()

    pygame.quit() 