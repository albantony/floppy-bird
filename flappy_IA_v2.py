import pygame
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import math

# Initialisation de pygame
pygame.init()
screen_width = 400
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Flappy Bird AI')
clock = pygame.time.Clock()

# Couleurs
WHITE = (255, 255, 255)
BLUE = (135, 206, 235)
GREEN = (0, 200, 0)

# Constantes du jeu
GRAVITY = 0.5
JUMP_STRENGTH = -10
PIPE_GAP = 150
PIPE_WIDTH = 70
BASE_HEIGHT = 80

# Classe oiseau
class Bird:
    def __init__(self):
        self.x = 50
        self.y = screen_height // 2
        self.velocity = 0
        self.radius = 20

    def jump(self):
        self.velocity = JUMP_STRENGTH

    def move(self):
        self.velocity += GRAVITY
        self.y += self.velocity

    def draw(self, screen):
        pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), self.radius)

# Classe tuyau
class Pipe:
    def __init__(self, x):
        self.x = x
        self.height = random.randint(100, 400)
        self.passed = False

    def move(self):
        self.x -= 5

    def draw(self, screen):
        # Tuyau du haut
        pygame.draw.rect(screen, GREEN, (self.x, 0, PIPE_WIDTH, self.height))
        # Tuyau du bas
        bottom_y = self.height + PIPE_GAP
        pygame.draw.rect(screen, GREEN, (self.x, bottom_y, PIPE_WIDTH, screen_height - bottom_y - BASE_HEIGHT))

    def collide(self, bird):
        bird_mask = pygame.Rect(bird.x - bird.radius, bird.y - bird.radius, bird.radius * 2, bird.radius * 2)
        top_pipe = pygame.Rect(self.x, 0, PIPE_WIDTH, self.height)
        bottom_pipe = pygame.Rect(self.x, self.height + PIPE_GAP, PIPE_WIDTH, screen_height - self.height - PIPE_GAP - BASE_HEIGHT)
        return bird_mask.colliderect(top_pipe) or bird_mask.colliderect(bottom_pipe)

# Base
class Base:
    def __init__(self):
        self.y = screen_height - BASE_HEIGHT

    def draw(self, screen):
        pygame.draw.rect(screen, (150, 75, 0), (0, self.y, screen_width, BASE_HEIGHT))

# Environnement
class FlappyBirdEnv:
    def reset(self):
        self.bird = Bird()
        self.pipes = [Pipe(300), Pipe(500)]
        self.base = Base()
        self.frame = 0
        self.score = 0
        return self.get_state()

    def step(self, action):
        reward = 0.1
        done = False

        if action == 1:
            self.bird.jump()

        self.bird.move()
        self.frame += 1

        remove = []
        add_pipe = False
        for pipe in self.pipes:
            pipe.move()
            if pipe.collide(self.bird):
                reward = -1
                done = True
            if pipe.x + PIPE_WIDTH < 0:
                remove.append(pipe)
            if not pipe.passed and pipe.x < self.bird.x:
                pipe.passed = True
                add_pipe = True
                reward = 1
                self.score += 1

        if add_pipe:
            self.pipes.append(Pipe(screen_width + 100))

        for r in remove:
            self.pipes.remove(r)

        if self.bird.y + self.bird.radius >= screen_height - BASE_HEIGHT or self.bird.y - self.bird.radius <= 0:
            reward = -1
            done = True

        return self.get_state(), reward, done

    def get_state(self):
        next_pipe = None
        for pipe in self.pipes:
            if pipe.x + PIPE_WIDTH > self.bird.x:
                next_pipe = pipe
                break
        if next_pipe is None:
            next_pipe = Pipe(screen_width + 100)

        return np.array([
            self.bird.y / screen_height,
            self.bird.velocity / 10,
            (next_pipe.x - self.bird.x) / screen_width,
            next_pipe.height / screen_height
        ], dtype=np.float32)

    def render(self):
        screen.fill(BLUE)
        self.base.draw(screen)
        for pipe in self.pipes:
            pipe.draw(screen)
        self.bird.draw(screen)
        pygame.display.update()
        clock.tick(60)

# Réseau neuronal
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.fc(x)

# Paramètres
num_episodes = 500
batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 1e-3
memory = deque(maxlen=2000)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

env = FlappyBirdEnv()

# Entraînement
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    SHOW_GAME = (episode % 50 == 0)

    while not done:
        if SHOW_GAME:
            env.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            with torch.no_grad():
                q_values = model(state_tensor)
                action = torch.argmax(q_values).item()

        next_state, reward, done = env.step(action)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        # Entraînement
        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

            state_batch = torch.tensor(np.array(state_batch), dtype=torch.float32).to(device)
            action_batch = torch.tensor(action_batch, dtype=torch.int64).unsqueeze(1).to(device)
            reward_batch = torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1).to(device)
            next_state_batch = torch.tensor(np.array(next_state_batch), dtype=torch.float32).to(device)
            done_batch = torch.tensor(done_batch, dtype=torch.float32).unsqueeze(1).to(device)

            q_values = model(state_batch).gather(1, action_batch)
            with torch.no_grad():
                next_q_values = model(next_state_batch).max(1)[0].unsqueeze(1)
                target_q_values = reward_batch + (gamma * next_q_values * (1 - done_batch))

            loss = loss_fn(q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {episode}, Total Reward: {round(total_reward, 2)}")

pygame.quit()
