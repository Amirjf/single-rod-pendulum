import math
import pygame

# Colors
white  = (255, 255, 255)
black  = (0, 0, 0)
gray   = (128, 128, 128)

blue   = (0, 0, 255)
red   = (255, 0, 0)
green   = (0, 255, 0)
yellow   = (255, 255, 0)
purple   = (255, 0, 255)
orange = (255, 165, 0)


class Pendulum:
    def __init__(self, pivot_x=0, pivot_y=0, m=1, l=300, 
                 a = math.pi/2, g=1, color=blue):
        self.pivot  = (pivot_x, pivot_y)
        self.m      = m
        self.l      = l
        self.a      = a
        self.g      = g
        self.color  = color
        self.color_traj  = self.mix_with_white()

        self.x      = 0
        self.y      = 0
        self.av     = 0

        self.trajectory = []  # Stores (x, y) positions of pendulum
        self.wave_x     = []  # Stores x-coordinates for sine wave

    def update(self):
        """ Update the pendulum's motion using physics equations """
        acc = (-self.g / self.l) * math.sin(self.a)  # Angular acceleration
        self.av += acc  # Angular velocity
        self.av *= 0.99  # Damping (simulates air resistance)

        self.a += self.av  # Angular position
        self.x = self.pivot[0] + self.l * math.sin(self.a)  # Update x
        self.y = self.pivot[1] + self.l * math.cos(self.a)  # Update y

        # Store x-coordinates for the sine wave trajectory
        if len(self.wave_x) > 400:  # Limit length of wave
            self.wave_x.pop(0)
        self.wave_x.append(self.x)

    def draw(self, surface):
        """ Draw pendulum (rod + mass) """
        pygame.draw.line(surface, white, self.pivot, (self.x, self.y), 2) # rod
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), 15) # mass

    def plot_trajectory(self, surface):
        """ Plot the previous positions of the pendulum """
        if len(self.trajectory) > 500:
            self.trajectory.pop(0)
        self.trajectory.append((self.x, self.y))

        for point in self.trajectory:
            pygame.draw.circle(surface, self.color_traj, (int(point[0]), int(point[1])), 2)

    def plot_sine_wave(self, surface, width, height, shift=0):
        """ Draw the sine wave at the bottom of the screen """
        if len(self.wave_x) > 1:
            amplif = 2
            for i in range(1, len(self.wave_x)):
                pygame.draw.line(surface, self.color, 
                                 (width - len(self.wave_x) + i - 1, height - 200 + shift + int(self.wave_x[i - 1] / 10 * amplif)), 
                                 (width - len(self.wave_x) + i, height - 200 + shift + int(self.wave_x[i] / 10 * amplif)), 2)

    def mix_with_white(self):
        factor = 0.5  # Mix factor
        # Ensure the factor is between 0 and 1
        factor = max(0, min(factor, 1))
        
        # The white color in RGB
        white = (255, 255, 255)
        
        # Mix the color with white using the factor
        mixed_color = tuple(int((1 - factor) * self.color[i] + factor * white[i]) for i in range(3))
        return mixed_color

def init_surface(size, caption):
    pygame.init()
    pygame.display.set_caption(caption)
    surface = pygame.display.set_mode(size)
    clock = pygame.time.Clock()
    return surface, clock

def run():
    """ Main loop to run the simulation """
    width, height = 800, 800
    fps = 60
    surface, clock = init_surface((width, height), "Pendulum with Sine Wave")

    lengths = [200+30*i for i in range(7)]
    colors = [red, blue, green, yellow, purple, orange, white]
    pend_list = [Pendulum(pivot_x=width//2, pivot_y=height//2, l=l, color=c)
                 for l, c in zip(lengths, colors)]
    pendulum = Pendulum(width//2, height//2)  # Start pivot at (width/2, 200)
    
    running = True
    while running:
        clock.tick(fps)
        surface.fill(black)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    running = False
                elif event.key == pygame.K_r:
                    pend_list = [Pendulum(pivot_x=width//2, pivot_y=height//2, l=l, color=c)
                                            for l, c in zip(lengths, colors)]

        # Update and draw the pendulum
        for i in range(len(pend_list)-1,-1,-1):
            pend_list[i].update()
            pend_list[i].plot_trajectory(surface)
            pend_list[i].draw(surface)
            pend_list[i].plot_sine_wave(surface, width, height, shift=-400)

        pygame.display.update()
    pygame.quit()

run()
