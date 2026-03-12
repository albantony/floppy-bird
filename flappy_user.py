import pygame
import random

pygame.init()
screen = pygame.display.set_mode((400, 600))
background = pygame.image.load("game/images/background.png").convert()
background = pygame.transform.scale(background, (400, 600))
screen_width, screen_height = screen.get_size()
image = pygame.image.load("game/images/Flappy.png").convert_alpha()
image = pygame.transform.scale(image, (50, 50))  # Resize the bird image
pygame.display.set_caption("Floppy Berk")

WHITE = (255, 255, 255)
GREEN = (0, 255, 0) 

class Bird:
    def __init__(self, image):
        self.image = image
        self.x = 100
        self.y = screen.get_height() // 2
        self.gravity = 0.5
        self.velocity = 0

    def jump(self):
        self.velocity = -10

    def update(self):
        self.velocity += self.gravity
        self.y += self.velocity

        if self.y > screen.get_height() - self.image.get_height():
            self.y = screen.get_height() - self.image.get_height()
            self.velocity = 0
        if self.y < 0:
            self.y = 0
            self.velocity = 0

    def draw(self, surface):
        surface.blit(self.image, (self.x, self.y))

    def get_rect(self):
        # Réduit la hitbox du bird pour plus de tolérance
        hitbox_width = int(self.image.get_width() * 0.7) 
        hitbox_height = int(self.image.get_height() * 0.7)
        offset_x = (self.image.get_width() - hitbox_width) // 2
        offset_y = (self.image.get_height() - hitbox_height) // 2
        return pygame.Rect(self.x + offset_x, self.y + offset_y, hitbox_width, hitbox_height)
    


class PipePair:
    def __init__(self):
        self.x = screen_width
        self.gap_size = 150
        self.gap_y = random.randint(100, screen_height - 100 - self.gap_size)
        self.width = 50
        self.speed = 3
        self.top_pipe = pygame.Rect(self.x, 0, self.width, self.gap_y - self.gap_size // 2)
        self.bottom_pipe = pygame.Rect(self.x, self.gap_y + self.gap_size // 2, self.width, screen_height - (self.gap_y + self.gap_size // 2))
        self.passed = False  # Ajouté pour le score


    def update(self):
        self.x -= self.speed
        self.top_pipe.x = self.x
        self.bottom_pipe.x = self.x

    def draw(self):
        pygame.draw.rect(screen, GREEN, self.top_pipe)
        pygame.draw.rect(screen, GREEN, self.bottom_pipe)

    def is_off_screen(self):
        return self.x < -self.width

    def collides_with(self, bird_rect):
        return bird_rect.colliderect(self.top_pipe) or bird_rect.colliderect(self.bottom_pipe)

bird = Bird(image)
pipes = [PipePair()]

running = True
clock = pygame.time.Clock()
spawn_pipe = pygame.USEREVENT
pygame.time.set_timer(spawn_pipe, 1500)
score = 0  # Ajouté pour le score
font = pygame.font.SysFont(None, 48)  # Police pour afficher le score
reward = 0 # Initialisation de la récompense
done = False  # Initialisation de l'état de fin de jeu

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            bird.jump()
        if event.type == spawn_pipe:
            pipes.append(PipePair())
            
    # Gestion du score
    for pipe in pipes:
        if not pipe.passed and pipe.x + pipe.width < bird.x:
            pipe.passed = True
            score += 1

    bird.update()
    for pipe in pipes:
        pipe.update()

    screen.fill(WHITE)
    screen.blit(background, (0 , 0)) 
    bird.draw(screen)
    for pipe in pipes:
        pipe.draw()
        
    # Affichage du score
    score_surface = font.render(str(score), True, (0, 0, 0))
    screen.blit(score_surface, (screen_width // 2 - score_surface.get_width() // 2, 30))
    
    # Trouver le prochain tuyau devant le bird
    next_pipes = [pipe for pipe in pipes if pipe.x + pipe.width > bird.x]
    if next_pipes:
        next_pipe = next_pipes[0]
        state = [
            bird.y,
            bird.velocity,
            next_pipe.x - bird.x,
            next_pipe.gap_y - next_pipe.gap_size // 2,
            next_pipe.gap_y + next_pipe.gap_size // 2
        ]
    else:
        # Valeurs par défaut si aucun tuyau devant
        state = [bird.y, bird.velocity, 0, screen_height // 2, screen_height // 2]
 
    bird_rect = bird.get_rect()
    for pipe in pipes:
        if pipe.collides_with(bird_rect):
            print("Game Over")
            print(f"Final Score: {score}")
            reward = score - 10 # Récompense négative pour collision
            done = True
            running = False


    pygame.display.flip()
    clock.tick(60)  

pygame.quit()
