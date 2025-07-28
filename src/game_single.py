import random
import pygame
import torch

from src.game_base import GameBase
from src.model import ANN


class Snake(GameBase):
    def __init__(
        self,
        parent_screen,
        length=5,
        color=(0, 255, 0),
        start_pos=None,
        model_path=None,
    ):
        super().__init__()
        self.length = length
        self.parent_screen = parent_screen
        self.color = color
        self.model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load model if path provided
        if model_path:
            self.model = ANN(16, 4).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()

        # Initialize snake at specified or random position
        if start_pos:
            start_x, start_y = start_pos
        else:
            start_x = random.randint(1, self.MAX_FOOD_INDEX - 1) * self.BLOCK_WIDTH
            start_y = random.randint(1, self.MAX_FOOD_INDEX - 1) * self.BLOCK_WIDTH
        self.x = [start_x] * self.length
        self.y = [start_y] * self.length
        self.direction = "right"
        self.is_stopped = False

    def draw(self):
        for i in range(self.length):
            pygame.draw.rect(
                self.parent_screen,
                self.color,
                (self.x[i], self.y[i], self.BLOCK_WIDTH, self.BLOCK_WIDTH),
            )

    def increase(self):
        self.length += 1
        self.x.append(-1)
        self.y.append(-1)

    def move_left(self):
        if self.direction != "right":
            self.direction = "left"

    def move_right(self):
        if self.direction != "left":
            self.direction = "right"

    def move_up(self):
        if self.direction != "down":
            self.direction = "up"

    def move_down(self):
        if self.direction != "up":
            self.direction = "down"

    def move(self):
        for i in range(self.length - 1, 0, -1):
            self.x[i] = self.x[i - 1]
            self.y[i] = self.y[i - 1]

        if self.direction == "right":
            self.x[0] += self.BLOCK_WIDTH
        if self.direction == "left":
            self.x[0] -= self.BLOCK_WIDTH
        if self.direction == "up":
            self.y[0] -= self.BLOCK_WIDTH
        if self.direction == "down":
            self.y[0] += self.BLOCK_WIDTH


class Apple(GameBase):
    def __init__(self, parent_screen):
        super().__init__()
        self.parent_screen = parent_screen
        self.x = 10 * self.BLOCK_WIDTH
        self.y = 10 * self.BLOCK_WIDTH

    def draw(self):
        pygame.draw.rect(
            self.parent_screen,
            (255, 0, 0),  # Red color for apple
            (self.x, self.y, self.BLOCK_WIDTH, self.BLOCK_WIDTH),
        )

    def spawn(self):
        self.x = random.randint(0, self.MAX_FOOD_INDEX) * self.BLOCK_WIDTH
        self.y = random.randint(0, self.MAX_FOOD_INDEX) * self.BLOCK_WIDTH


class Game(GameBase):
    def __init__(self, model_path=None):
        super().__init__()
        pygame.init()
        pygame.display.set_caption("Snake Game - Single Player")
        self.SCREEN_UPDATE = pygame.USEREVENT
        self.timer = 1
        pygame.time.set_timer(self.SCREEN_UPDATE, self.timer)
        self.surface = pygame.display.set_mode((self.SCREEN_SIZE, self.SCREEN_SIZE))

        # Initialize single snake
        self.snake = Snake(
            self.surface,
            length=1,
            color=(0, 255, 0),
            model_path=model_path,
        )

        self.apple = Apple(self.surface)
        self.score = 0
        self.game_over = False
        self.reward = 0
        self.current_distance_food = float("inf")

    def play(self):
        pygame.time.set_timer(self.SCREEN_UPDATE, self.timer)

        # Clear screen
        self.surface.fill((0, 0, 0))

        # Draw snake and apple
        self.snake.draw()
        self.apple.draw()

        # Update display
        pygame.display.flip()

    def reset(self):
        self.snake = Snake(self.surface, length=1)
        self.apple = Apple(self.surface)
        self.score = 0
        self.game_over = False

    def is_danger(self, position_checking):
        x, y = position_checking
        # Check if position is out of bounds
        if x < 0 or x >= self.SCREEN_SIZE or y < 0 or y >= self.SCREEN_SIZE:
            return True
        # Check if position collides with snake body
        for i in range(1, self.snake.length):
            if self.snake.x[i] == x and self.snake.y[i] == y:
                return True
        return False

    def is_collision(self, x1, y1, x2, y2):
        return x1 == x2 and y1 == y2

    def run(self, action):
        self.reward = 0
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.snake.move_up()
                if event.key == pygame.K_DOWN:
                    self.snake.move_down()
                if event.key == pygame.K_LEFT:
                    self.snake.move_left()
                if event.key == pygame.K_RIGHT:
                    self.snake.move_right()
            elif event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Apply action if provided (for AI training)
        if action is not None:
            if action[0] == 1:
                self.snake.move_up()
            elif action[1] == 1:
                self.snake.move_down()
            elif action[2] == 1:
                self.snake.move_left()
            elif action[3] == 1:
                self.snake.move_right()

        self.snake.move()

        # Check collision with apple
        if self.is_collision(
            self.snake.x[0], self.snake.y[0], self.apple.x, self.apple.y
        ):
            self.apple.spawn()
            self.snake.increase()
            self.score += 1
            self.reward = 10

        # Check collision with boundaries
        if (
            self.snake.x[0] < 0
            or self.snake.x[0] >= self.SCREEN_SIZE
            or self.snake.y[0] < 0
            or self.snake.y[0] >= self.SCREEN_SIZE
        ):
            self.game_over = True
            self.reward = -100
            return self.reward, self.game_over, self.score

        # Check collision with itself
        for i in range(1, self.snake.length):
            if self.is_collision(
                self.snake.x[0], self.snake.y[0], self.snake.x[i], self.snake.y[i]
            ):
                self.game_over = True
                self.reward = -100
                return self.reward, self.game_over, self.score

        # Calculate reward based on distance to apple
        if self.reward == 0:
            current_distance = abs(self.snake.x[0] - self.apple.x) + abs(
                self.snake.y[0] - self.apple.y
            )
            # Base reward for each step
            self.reward = -0.1
            # Additional reward based on distance to apple
            if current_distance < self.BLOCK_WIDTH * 2:  # Very close to apple
                self.reward += 5
            elif current_distance < self.BLOCK_WIDTH * 4:  # Moderately close to apple
                self.reward += 2
            elif current_distance > self.BLOCK_WIDTH * 6:  # Far from apple
                self.reward -= 2

        self.play()
        return self.reward, self.game_over, self.score
