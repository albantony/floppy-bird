import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

class Bird:
    """Classe représentant l'oiseau dans le jeu Flappy Bird.
    Elle gère la position, la gravité, les sauts et le dessin de l'oiseau."""
    def __init__(self, image, screen_height):
        # définit les attributs de l'oiseau
        self.image = image
        self.x = 100
        self.y = screen_height // 2
        self.gravity = 0.5
        self.velocity = 0
        self.screen_height = screen_height

    def jump(self):
        #fonction pour faire sauter l'oiseau
        self.velocity = -10

    def update(self):
        # met à jour la position de l'oiseau en fonction de la gravité frame par frame
        self.velocity += self.gravity
        self.y += self.velocity

        if self.y > self.screen_height - self.image.get_height():
            self.y = self.screen_height - self.image.get_height()
            self.velocity = 0
        if self.y < 0:
            self.y = 0
            self.velocity = 0

    def draw(self, surface):
        # dessine l'oiseau sur la surface donnée
        surface.blit(self.image, (self.x, self.y))

    def get_rect(self):
        # associe une hitbox à l'oiseau pour gérer les collisions
        hitbox_width = int(self.image.get_width() * 0.7)
        hitbox_height = int(self.image.get_height() * 0.7)
        offset_x = (self.image.get_width() - hitbox_width) // 2
        offset_y = (self.image.get_height() - hitbox_height) // 2
        return pygame.Rect(self.x + offset_x, self.y + offset_y, hitbox_width, hitbox_height)


class PipePair:
    """Classe représentant une paire de tuyaux dans le jeu Flappy Bird.
    Elle gère la position, la taille des tuyaux, leur mouvement et les collisions."""
    def __init__(self, screen_width, screen_height):
        self.x = screen_width
        self.gap_size = 150
        self.gap_y = random.randint(100, screen_height - 100 - self.gap_size)
        self.width = 50
        self.speed = 3
        self.screen_height = screen_height
        self.passed = False
        self.top_pipe = pygame.Rect(self.x, 0, self.width, self.gap_y - self.gap_size // 2)
        self.bottom_pipe = pygame.Rect(self.x, self.gap_y + self.gap_size // 2, self.width,
                                       screen_height - (self.gap_y + self.gap_size // 2))

    def update(self):
        self.x -= self.speed
        self.top_pipe.x = self.x
        self.bottom_pipe.x = self.x

    def draw(self, screen):
        pygame.draw.rect(screen, GREEN, self.top_pipe)
        pygame.draw.rect(screen, GREEN, self.bottom_pipe)

    def is_off_screen(self):
        return self.x < -self.width

    def collides_with(self, bird_rect):
        return bird_rect.colliderect(self.top_pipe) or bird_rect.colliderect(self.bottom_pipe)


class FlappyEnv:
    """Classe représentant l'environnement de jeu Flappy Bird.
    Elle gère l'initialisation de Pygame, le dessin des éléments du jeu"""
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((400, 600))
        pygame.display.set_caption("FlappyEnv")
        self.clock = pygame.time.Clock()
        self.background = pygame.transform.scale(pygame.image.load("game/images/background.png").convert(), (400, 600))
        self.bird_image = pygame.transform.scale(pygame.image.load("game/images/Flappy.png").convert_alpha(), (50, 50))
        self.font = pygame.font.SysFont(None, 36)

        self.screen_width, self.screen_height = self.screen.get_size()
        self.reset()

    def reset(self):
        """Réinitialise l'environnement pour un nouvel épisode de jeu."""
        self.bird = Bird(self.bird_image, self.screen_height)
        self.pipes = [PipePair(self.screen_width, self.screen_height)]
        self.score = 0
        self.frame_count = 0
        self.done = False
        return self.get_state()

    def get_state(self):
        """Retourne l'état actuel de l'environnement sous forme de tableau numpy."""
        # État simplifié : [bird.y, bird.velocity, next_pipe_x, next_pipe_gap_y]
        next_pipe = None
        for pipe in self.pipes:
            if pipe.x + pipe.width > self.bird.x:
                next_pipe = pipe
                break
        if not next_pipe:
            next_pipe = PipePair(self.screen_width, self.screen_height)  # pipe factice

        state = np.array([
            self.bird.y / self.screen_height,
            self.bird.velocity / 10.0,
            (next_pipe.x - self.bird.x) / self.screen_width,
            next_pipe.gap_y / self.screen_height
        ], dtype=np.float32)
        return state

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True, {}

        # Action: 0 = rien, 1 = saut
        if action == 1:
            self.bird.jump()

        self.bird.update()
        reward = 0.1  # petit reward par frame de survie
        self.frame_count += 1

        for pipe in self.pipes:
            pipe.update()
            if not pipe.passed and pipe.x + pipe.width < self.bird.x:
                pipe.passed = True
                self.score += 1
                reward += 1  # reward pour un pipe passé

        # Supprimer les tuyaux hors écran
        self.pipes = [pipe for pipe in self.pipes if not pipe.is_off_screen()]
        if self.frame_count % 90 == 0:
            self.pipes.append(PipePair(self.screen_width, self.screen_height))

        # Collision
        bird_rect = self.bird.get_rect()
        for pipe in self.pipes:
            if pipe.collides_with(bird_rect):
                self.done = True
                reward = -5
                break

        if self.bird.y >= self.screen_height - self.bird.image.get_height():
            self.done = True
            reward = -5

        return self.get_state(), reward, self.done, {}

    def render(self):
        self.screen.fill(WHITE)
        self.screen.blit(self.background, (0, 0))
        self.bird.draw(self.screen)
        for pipe in self.pipes:
            pipe.draw(self.screen)

        score_surface = self.font.render(str(self.score), True, (0, 0, 0))
        self.screen.blit(score_surface, (self.screen_width // 2 - score_surface.get_width() // 2, 20))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()
        

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=2):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def train_dqn(model, target_model, optimizer, memory, batch_size=64, gamma=0.99):
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*transitions)

    state_batch = torch.tensor(state_batch, dtype=torch.float32)
    action_batch = torch.tensor(action_batch, dtype=torch.int64).unsqueeze(1)
    reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
    next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32)
    done_batch = torch.tensor(done_batch, dtype=torch.float32)

    q_values = model(state_batch).gather(1, action_batch).squeeze()
    next_q_values = target_model(next_state_batch).max(1)[0]
    target = reward_batch + gamma * next_q_values * (1 - done_batch)

    loss = nn.MSELoss()(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# === ENTRAÎNEMENT ===
env = FlappyEnv()
num_episodes = 500
memory = ReplayMemory(10000)
model = DQN(input_size=4)
target_model = DQN(input_size=4)
target_model.load_state_dict(model.state_dict())
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Choix de l'action : exploration ou exploitation
        if random.random() < epsilon:
            action = random.choice([0, 1])
        else:
            with torch.no_grad():
                action = model(torch.tensor(state, dtype=torch.float32)).argmax().item()

        next_state, reward, done, _ = env.step(action)
        memory.push((state, action, reward, next_state, done))

        train_dqn(model, target_model, optimizer, memory)
        state = next_state
        total_reward += reward

        # Optionnel : afficher le jeu pendant l'entraînement
        env.render()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if episode % 10 == 0:
        target_model.load_state_dict(model.state_dict())
        print(f"Episode {episode}, Total Reward: {total_reward:.2f}")

env.close()
