import random
import time

import numpy as np
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
        self.direction = "left"

    def move_right(self):
        self.direction = "right"

    def move_up(self):
        self.direction = "up"

    def move_down(self):
        self.direction = "down"

    def move(self):
        if self.is_stopped:
            return

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

        self.draw()


class Apple(GameBase):
    def __init__(self, parent_screen):
        super().__init__()
        self.parent_screen = parent_screen
        self.x = random.randint(0, self.MAX_FOOD_INDEX) * self.BLOCK_WIDTH
        self.y = random.randint(0, self.MAX_FOOD_INDEX) * self.BLOCK_WIDTH

    def draw(self):
        pygame.draw.rect(
            self.parent_screen,
            (255, 0, 0),
            (self.x, self.y, self.BLOCK_WIDTH, self.BLOCK_WIDTH),
        )

    def move(self, snakes):
        while True:  # make sure new food is not getting created over snake body
            x = random.randint(0, self.MAX_FOOD_INDEX) * self.BLOCK_WIDTH
            y = random.randint(0, self.MAX_FOOD_INDEX) * self.BLOCK_WIDTH
            clean = True
            for snake in snakes:
                for i in range(0, snake.length):
                    if x == snake.x[i] and y == snake.y[i]:
                        clean = False
                        break
                if not clean:
                    break
            if clean:
                self.x = x
                self.y = y
                return


class Game(GameBase):
    def __init__(self, model_path1=None, model_path2=None):
        super().__init__()
        pygame.init()
        pygame.display.set_caption("Snake Game - 2 Player Competition")
        self.SCREEN_UPDATE = pygame.USEREVENT
        self.timer = 1
        pygame.time.set_timer(self.SCREEN_UPDATE, self.timer)
        self.surface = pygame.display.set_mode((self.SCREEN_SIZE, self.SCREEN_SIZE))

        # Initialize two snakes with different colors and positions
        self.snake1 = Snake(
            self.surface,
            length=1,
            color=(0, 255, 0),
            start_pos=(self.BLOCK_WIDTH * 5, self.BLOCK_WIDTH * 10),
            model_path=model_path1,
        )  # Green snake
        self.snake2 = Snake(
            self.surface,
            length=1,
            color=(0, 0, 255),
            start_pos=(self.BLOCK_WIDTH * 15, self.BLOCK_WIDTH * 10),
            model_path=model_path2,
        )  # Blue snake

        self.snakes = [self.snake1, self.snake2]
        self.apple = None
        self.apple_spawn_time = time.time()
        self.apple_lifetime = 5.0  # Apple exists for 5 seconds
        self.scores = [0, 0]  # Scores for snake1 and snake2
        self.game_over = False
        self.rewards = [0, 0]  # Rewards for each snake
        self.current_distance_food = float("inf")

    def play(self):
        pygame.time.set_timer(self.SCREEN_UPDATE, self.timer)

        # Clear screen
        self.surface.fill((0, 0, 0))

        # Handle apple spawning/despawning
        current_time = time.time()
        if self.apple is None:
            # No apple exists, spawn one and stop snakes
            if current_time - self.apple_spawn_time >= 5.0:
                self.apple = Apple(parent_screen=self.surface)
                self.apple.move(self.snakes)
                self.apple_spawn_time = current_time
                # Resume snake movement
                for snake in self.snakes:
                    snake.is_stopped = False
        else:
            # Apple exists, check if it should despawn
            if current_time - self.apple_spawn_time >= self.apple_lifetime:
                self.apple = None
                # Stop snake movement
                for snake in self.snakes:
                    snake.is_stopped = True

        # Move snakes and draw them
        for snake in self.snakes:
            snake.move()

        # Draw apple if it exists
        if self.apple:
            self.apple.draw()

        self.display_score()

        # Calculate rewards for each snake
        for i, snake in enumerate(self.snakes):
            self.rewards[i] = self.reward_function_for_snake(i)

    def is_collision(self, snake_idx):
        snake = self.snakes[snake_idx]
        head_x = snake.x[0]
        head_y = snake.y[0]

        # Check self collision
        for i in range(1, snake.length):
            if head_x == snake.x[i] and head_y == snake.y[i]:
                return True

        # Check collision with other snake
        other_snake = self.snakes[1 - snake_idx]
        for i in range(other_snake.length):
            if head_x == other_snake.x[i] and head_y == other_snake.y[i]:
                return True

        # Check wall collision
        if (
            head_x > (self.SCREEN_SIZE - self.BLOCK_WIDTH)
            or head_y > (self.SCREEN_SIZE - self.BLOCK_WIDTH)
            or head_x < 0
            or head_y < 0
        ):
            return True

        return False

    def is_danger(self, point, snake_idx):
        point_x = point[0]
        point_y = point[1]

        # Check collision with self
        snake = self.snakes[snake_idx]
        for i in range(1, snake.length):
            if point_x == snake.x[i] and point_y == snake.y[i]:
                return True

        # Check collision with other snake
        other_snake = self.snakes[1 - snake_idx]
        for i in range(other_snake.length):
            if point_x == other_snake.x[i] and point_y == other_snake.y[i]:
                return True

        # Check wall collision
        if (
            point_x > (self.SCREEN_SIZE - self.BLOCK_WIDTH)
            or point_y > (self.SCREEN_SIZE - self.BLOCK_WIDTH)
            or point_x < 0
            or point_y < 0
        ):
            return True

        return False

    def display_score(self):
        font = pygame.font.SysFont("arial", 20)
        # Display score for snake 1 (green)
        msg1 = f"Green: {self.scores[0]}"
        score1 = font.render(msg1, True, (0, 255, 0))
        self.surface.blit(score1, (10, 10))

        # Display score for snake 2 (blue)
        msg2 = f"Blue: {self.scores[1]}"
        score2 = font.render(msg2, True, (0, 0, 255))
        self.surface.blit(score2, (480, 10))

    def reset(self):
        # Reset both snakes
        self.snake1 = Snake(
            self.surface,
            length=1,
            color=(0, 255, 0),
            start_pos=(self.BLOCK_WIDTH * 5, self.BLOCK_WIDTH * 10),
            model_path=(
                self.snake1.model_path if hasattr(self.snake1, "model_path") else None
            ),
        )
        self.snake2 = Snake(
            self.surface,
            length=1,
            color=(0, 0, 255),
            start_pos=(self.BLOCK_WIDTH * 15, self.BLOCK_WIDTH * 10),
            model_path=(
                self.snake2.model_path if hasattr(self.snake2, "model_path") else None
            ),
        )
        self.snakes = [self.snake1, self.snake2]
        self.apple = None
        self.apple_spawn_time = time.time()
        self.scores = [0, 0]
        self.game_over = False

    def get_next_direction(self, move):
        # ["right", "down", "left", "up"]
        new_dir = "right"
        if np.array_equal(move, [1, 0, 0, 0]):
            new_dir = "right"
        if np.array_equal(move, [0, 1, 0, 0]):
            new_dir = "down"
        if np.array_equal(move, [0, 0, 1, 0]):
            new_dir = "left"
        if np.array_equal(move, [0, 0, 0, 1]):
            new_dir = "up"

        return new_dir

    def reward_function_for_snake(self, snake_idx):
        reward = 0
        snake = self.snakes[snake_idx]

        # Check if snake ate apple
        if self.apple and snake.x[0] == self.apple.x and snake.y[0] == self.apple.y:
            reward = 10
            self.scores[snake_idx] += 1
            snake.increase()
            self.apple.move(self.snakes)
            self.apple_spawn_time = time.time()

        # Check collision
        if self.is_collision(snake_idx):
            reward = -10
            self.game_over = True

        # Distance-based reward if apple exists
        if self.apple and reward == 0:
            dist_after_move = np.linalg.norm(
                np.array([snake.x[0], snake.y[0]])
                - np.array([self.apple.x, self.apple.y])
            )
            if dist_after_move < self.current_distance_food:
                reward = 1
            else:
                reward = -1
            self.current_distance_food = dist_after_move

        return reward

    def get_state(self, snake_idx):
        snake = self.snakes[snake_idx]
        other_snake = self.snakes[1 - snake_idx]
        head = [snake.x[0], snake.y[0]]

        # Danger straight, right, left
        point_l = [head[0] - self.BLOCK_WIDTH, head[1]]
        point_r = [head[0] + self.BLOCK_WIDTH, head[1]]
        point_u = [head[0], head[1] - self.BLOCK_WIDTH]
        point_d = [head[0], head[1] + self.BLOCK_WIDTH]

        dir_l = snake.direction == "left"
        dir_r = snake.direction == "right"
        dir_u = snake.direction == "up"
        dir_d = snake.direction == "down"

        state = [
            # Danger straight
            (dir_r and self.is_danger(point_r, snake_idx))
            or (dir_l and self.is_danger(point_l, snake_idx))
            or (dir_u and self.is_danger(point_u, snake_idx))
            or (dir_d and self.is_danger(point_d, snake_idx)),
            # Danger right
            (dir_u and self.is_danger(point_r, snake_idx))
            or (dir_d and self.is_danger(point_l, snake_idx))
            or (dir_l and self.is_danger(point_u, snake_idx))
            or (dir_r and self.is_danger(point_d, snake_idx)),
            # Danger left
            (dir_d and self.is_danger(point_r, snake_idx))
            or (dir_u and self.is_danger(point_l, snake_idx))
            or (dir_r and self.is_danger(point_u, snake_idx))
            or (dir_l and self.is_danger(point_d, snake_idx)),
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location relative to head (if apple exists)
            self.apple.x < head[0] if self.apple else 0,  # food left
            self.apple.x > head[0] if self.apple else 0,  # food right
            self.apple.y < head[1] if self.apple else 0,  # food up
            self.apple.y > head[1] if self.apple else 0,  # food down
            # Other snake head relative position
            other_snake.x[0] < head[0],  # other snake left
            other_snake.x[0] > head[0],  # other snake right
            other_snake.y[0] < head[1],  # other snake up
            other_snake.y[0] > head[1],  # other snake down
            # Apple exists
            self.apple is not None,
        ]

        return np.array(state, dtype=int)

    def run(self):
        # Get AI decisions for both snakes if they have models
        moves = []
        for i, snake in enumerate(self.snakes):
            if snake.model:
                state = (
                    torch.FloatTensor(self.get_state(i)).unsqueeze(0).to(snake.device)
                )
                with torch.no_grad():
                    prediction = snake.model(state)
                move = torch.argmax(prediction).item()
                # Convert to one-hot encoding
                move_array = [0, 0, 0, 0]
                move_array[move] = 1
                moves.append(move_array)
            else:
                # Default move (right)
                moves.append([1, 0, 0, 0])

        # Apply moves to snakes
        for i, (snake, move) in enumerate(zip(self.snakes, moves)):
            dir = self.get_next_direction(move)
            if dir == "left":
                snake.move_left()
            elif dir == "right":
                snake.move_right()
            elif dir == "down":
                snake.move_down()
            elif dir == "up":
                snake.move_up()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == self.SCREEN_UPDATE:
                self.play()
                pygame.display.update()
                pygame.time.Clock().tick(200)
                break

        return self.rewards, self.game_over, self.scores
