# Simple pygame program

# PARAMETERS ---------------------
isFullScreenActive = True


black = (0, 0, 0)
white = (255, 255, 255)


# Import and initialize the pygame library
import pygame

def init_surface(size,caption):
    pygame.init()
    pygame.display.set_caption(caption)
    
    # Set up the drawing window - in this case set the size of the window
    screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])

    surface = pygame.display.set_mode(size)
    clock = pygame.time.Clock()
    return surface, clock

def run_game(surface, clock, fps):
    if isFullScreenActive:
        # Get the size of the screen
        screen_info = pygame.display.Info()
        SCREEN_WIDTH, SCREEN_HEIGHT = screen_info.current_w, screen_info.current_h
    else:
        SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600

    size = (SCREEN_WIDTH, SCREEN_HEIGHT)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        pygame.display.update()
        clock.tick(fps)
    pygame.quit()



player = pygame.Rect((50, 50, 50, 50))

# Run until the user asks to quit
running = True
while running:
    
    # Fill the background with white
    screen.fill((255, 255, 255))

    # Draw a solid blue circle in the center
    pygame.draw.circle(screen, (0, 0, 255), (125, 250), 100)
    
    pygame.draw.rect(screen, (255, 0, 255), player)

    # Move the player
    key = pygame.key.get_pressed()
    if key[pygame.K_LEFT]:
        player.x -= 1
    if key[pygame.K_RIGHT]:
        player.x += 1
    if key[pygame.K_UP]:
        player.y -= 1
    if key[pygame.K_DOWN]:
        player.y += 1   

    # event handling, gets all event from the event queue
    for event in pygame.event.get(): # Did the user click the window close button?
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False

    # Flip the display - This line updates the entire screen with the new drawings. It effectively swaps the buffers, displaying what has been drawn since the last flip.
    pygame.display.update()

# Done! Time to quit.
pygame.quit()

# Run the game
run_game(surface, clock, 60)