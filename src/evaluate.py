import argparse
import time
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.game_base import GameBase
from src.model import ANN  # Assuming ANN is defined in model.py


def load_model(model_path, device, state_size=16, action_size=4):
    """Dynamically load a model based on its architecture"""
    model = ANN(state_size=state_size, action_size=action_size)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from {model_path} and moved to {device}")
    return model


class CompetitiveSnake(GameBase):
    def __init__(self, model, snake_id=1, start_pos=None, color=(0, 255, 0)):
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device
        self.snake_id = snake_id
        self.color = color
        self.length = 5
        self.score = 0
        self.steps = 0
        self.paused = False
        self.frozen = False
        self.freeze_counter = 0

        # Initialize position
        if start_pos:
            start_x, start_y = start_pos
        else:
            # Default positions for two snakes - keep away from edges
            if snake_id == 1:
                start_x, start_y = 10 * self.BLOCK_WIDTH, 10 * self.BLOCK_WIDTH
            else:
                start_x, start_y = 20 * self.BLOCK_WIDTH, 20 * self.BLOCK_WIDTH

        self.x = [start_x] * self.length
        self.y = [start_y] * self.length
        self.direction = "right" if snake_id == 1 else "left"

    def get_state(self, apple_pos, other_snake):
        """Get state vector for the neural network"""
        head_x, head_y = self.x[0], self.y[0]

        # Danger detection (walls, self collision, other snake)
        danger = []
        directions = [
            (-self.BLOCK_WIDTH, 0),  # left
            (self.BLOCK_WIDTH, 0),  # right
            (0, -self.BLOCK_WIDTH),  # up
            (0, self.BLOCK_WIDTH),  # down
            (-self.BLOCK_WIDTH, -self.BLOCK_WIDTH),  # up-left
            (self.BLOCK_WIDTH, -self.BLOCK_WIDTH),  # up-right
            (-self.BLOCK_WIDTH, self.BLOCK_WIDTH),  # down-left
            (self.BLOCK_WIDTH, self.BLOCK_WIDTH),  # down-right
        ]

        for dx, dy in directions:
            new_x, new_y = head_x + dx, head_y + dy

            # Check wall collision
            if (
                new_x < 0
                or new_x >= self.SCREEN_SIZE
                or new_y < 0
                or new_y >= self.SCREEN_SIZE
            ):
                danger.append(1)
            # Check self collision
            elif any(
                new_x == self.x[i] and new_y == self.y[i] for i in range(1, self.length)
            ):
                danger.append(1)
            # Check other snake collision
            elif other_snake and any(
                new_x == other_snake.x[i] and new_y == other_snake.y[i]
                for i in range(other_snake.length)
            ):
                danger.append(1)
            else:
                danger.append(0)

        # Current direction (one-hot)
        dir_up = 1 if self.direction == "up" else 0
        dir_down = 1 if self.direction == "down" else 0
        dir_left = 1 if self.direction == "left" else 0
        dir_right = 1 if self.direction == "right" else 0

        # Apple position relative to head
        apple_x, apple_y = apple_pos if apple_pos else (0, 0)
        apple_left = 1 if apple_x < head_x else 0
        apple_right = 1 if apple_x > head_x else 0
        apple_up = 1 if apple_y < head_y else 0
        apple_down = 1 if apple_y > head_y else 0

        state = danger + [
            dir_up,
            dir_down,
            dir_left,
            dir_right,
            apple_left,
            apple_right,
            apple_up,
            apple_down,
        ]

        return torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    def get_action(self, state):
        """Get action from model"""
        # For testing: use random actions with some bias towards apple
        if np.random.random() < 0.1:  # 10% random for exploration
            return np.random.randint(0, 4)
        
        with torch.no_grad():
            q_values = self.model(state.to(self.device))
            action = torch.argmax(q_values).item()
        return action

    def move(self, action):
        """Move snake based on action"""
        if self.paused or self.frozen:
            return

        # Map action to direction
        action_map = {0: "up", 1: "down", 2: "left", 3: "right"}
        new_direction = action_map[action]

        # Prevent 180-degree turns
        invalid_moves = {"up": "down", "down": "up", "left": "right", "right": "left"}
        if (
            self.direction in invalid_moves
            and new_direction == invalid_moves[self.direction]
        ):
            new_direction = self.direction

        self.direction = new_direction

        # Update position
        head_x, head_y = self.x[0], self.y[0]

        if self.direction == "up":
            head_y -= self.BLOCK_WIDTH
        elif self.direction == "down":
            head_y += self.BLOCK_WIDTH
        elif self.direction == "left":
            head_x -= self.BLOCK_WIDTH
        elif self.direction == "right":
            head_x += self.BLOCK_WIDTH

        # Move body
        self.x = [head_x] + self.x[:-1]
        self.y = [head_y] + self.y[:-1]
        self.steps += 1

    def check_collision(self, other_snake=None):
        """Check if snake collided with walls, itself, or other snake"""
        head_x, head_y = self.x[0], self.y[0]

        # Wall collision
        if (
            head_x < 0
            or head_x >= self.SCREEN_SIZE
            or head_y < 0
            or head_y >= self.SCREEN_SIZE
        ):
            return True

        # Self collision
        if any(
            head_x == self.x[i] and head_y == self.y[i] for i in range(1, self.length)
        ):
            return True

        # Other snake collision - skip checking because snakes can overlap
        # if other_snake:
        #     if any(
        #         head_x == other_snake.x[i] and head_y == other_snake.y[i]
        #         for i in range(other_snake.length)
        #     ):
        #         return True

        return False

    def check_apple_collision(self, apple_pos):
        """Check if snake ate the apple"""
        if not apple_pos:
            return False
        return self.x[0] == apple_pos[0] and self.y[0] == apple_pos[1]

    def grow(self):
        """Increase snake length"""
        self.length += 1
        self.x.append(self.x[-1])
        self.y.append(self.y[-1])
        self.score += 1

    def draw(self, surface):
        """Draw snake on surface"""
        for i in range(self.length):
            pygame.draw.rect(
                surface,
                self.color,
                (self.x[i], self.y[i], self.BLOCK_WIDTH, self.BLOCK_WIDTH),
            )


class CompetitiveGame(GameBase):
    def __init__(
        self,
        model1_path,
        model2_path,
        apple_freq=100,
        freeze_time=30,
        display=True,
        state_size=16,
        action_size=4,
    ):
        super().__init__()
        self.display = display
        self.apple_freq = apple_freq
        self.freeze_time = freeze_time
        self.steps_since_apple = 0
        self.apple_pos = None
        self.game_over = False

        # Load models
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model1 = load_model(
            model1_path, device, state_size=state_size, action_size=action_size
        )
        model2 = load_model(
            model2_path, device, state_size=state_size, action_size=action_size
        )

        # Create snakes with safer initial positions
        self.snake1 = CompetitiveSnake(model1, snake_id=1, color=(0, 255, 0))
        self.snake2 = CompetitiveSnake(model2, snake_id=2, color=(0, 0, 255))
        
        if not self.display:
            print(f"Snake1 initial position: {self.snake1.x[0]}, {self.snake1.y[0]}")
            print(f"Snake2 initial position: {self.snake2.x[0]}, {self.snake2.y[0]}")
        
        # Spawn initial apple
        self.spawn_apple()
        if not self.display:
            print(f"Initial apple at: {self.apple_pos}")

        # Initialize pygame if display is enabled
        if self.display:
            pygame.init()
            self.surface = pygame.display.set_mode((self.SCREEN_SIZE, self.SCREEN_SIZE))
            pygame.display.set_caption("Snake Competition")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

    def spawn_apple(self):
        """Spawn a new apple at random position"""
        while True:
            x = (
                np.random.randint(0, self.SCREEN_SIZE // self.BLOCK_WIDTH)
                * self.BLOCK_WIDTH
            )
            y = (
                np.random.randint(0, self.SCREEN_SIZE // self.BLOCK_WIDTH)
                * self.BLOCK_WIDTH
            )

            # Make sure apple doesn't spawn on snakes
            if not any(
                x == self.snake1.x[i] and y == self.snake1.y[i]
                for i in range(self.snake1.length)
            ):
                if not any(
                    x == self.snake2.x[i] and y == self.snake2.y[i]
                    for i in range(self.snake2.length)
                ):
                    self.apple_pos = (x, y)
                    break

    def step(self):
        """Execute one game step"""
        # Update freeze counters
        if self.snake1.frozen:
            self.snake1.freeze_counter -= 1
            if self.snake1.freeze_counter <= 0:
                self.snake1.frozen = False
                if not self.display:
                    print(f"Snake1 unfrozen")
        
        if self.snake2.frozen:
            self.snake2.freeze_counter -= 1
            if self.snake2.freeze_counter <= 0:
                self.snake2.frozen = False
                if not self.display:
                    print(f"Snake2 unfrozen")
        
        # Spawn apple if needed
        if self.steps_since_apple % self.apple_freq == 0 and self.apple_pos is None:
            self.spawn_apple()
            self.snake1.paused = False
            self.snake2.paused = False

        # Get states and actions
        state1 = self.snake1.get_state(self.apple_pos, self.snake2)
        state2 = self.snake2.get_state(self.apple_pos, self.snake1)

        action1 = self.snake1.get_action(state1)
        action2 = self.snake2.get_action(state2)

        # Move snakes
        if not self.display and self.snake1.steps == 0:
            print(f"Snake1 action: {action1}, direction: {self.snake1.direction}")
            print(f"Snake2 action: {action2}, direction: {self.snake2.direction}")
        
        self.snake1.move(action1)
        self.snake2.move(action2)
        
        if not self.display and self.snake1.steps == 1:
            print(f"Snake1 new position: {self.snake1.x[0]}, {self.snake1.y[0]}")
            print(f"Snake2 new position: {self.snake2.x[0]}, {self.snake2.y[0]}")

        # Check collisions with walls/self
        snake1_collision = self.snake1.check_collision()
        snake2_collision = self.snake2.check_collision()

        # Freeze snakes instead of ending game
        if snake1_collision and not self.snake1.frozen:
            self.snake1.frozen = True
            self.snake1.freeze_counter = self.freeze_time
            # Reset position to safe location
            self.snake1.x = [10 * self.BLOCK_WIDTH] * self.snake1.length
            self.snake1.y = [10 * self.BLOCK_WIDTH] * self.snake1.length
            self.snake1.direction = "right"
            if not self.display:
                print(f"Snake1 frozen for {self.freeze_time} steps")
        
        if snake2_collision and not self.snake2.frozen:
            self.snake2.frozen = True
            self.snake2.freeze_counter = self.freeze_time
            # Reset position to safe location
            self.snake2.x = [20 * self.BLOCK_WIDTH] * self.snake2.length
            self.snake2.y = [20 * self.BLOCK_WIDTH] * self.snake2.length
            self.snake2.direction = "left"
            if not self.display:
                print(f"Snake2 frozen for {self.freeze_time} steps")

        # Check apple collisions
        snake1_ate = self.snake1.check_apple_collision(self.apple_pos)
        snake2_ate = self.snake2.check_apple_collision(self.apple_pos)

        if snake1_ate or snake2_ate:
            # Model 1 has priority if both eat simultaneously
            if snake1_ate:
                self.snake1.grow()
                if not self.display:
                    print(f"Snake1 ate apple! Score: {self.snake1.score}")
            elif snake2_ate:
                self.snake2.grow()
                if not self.display:
                    print(f"Snake2 ate apple! Score: {self.snake2.score}")

            # Remove apple and pause snakes
            self.apple_pos = None
            self.snake1.paused = True
            self.snake2.paused = True
            self.steps_since_apple = 0

        self.steps_since_apple += 1

    def render(self):
        """Render the game"""
        if not self.display:
            return

        # Clear screen
        self.surface.fill((0, 0, 0))

        # Draw apple
        if self.apple_pos:
            pygame.draw.rect(
                self.surface,
                (255, 0, 0),
                (
                    self.apple_pos[0],
                    self.apple_pos[1],
                    self.BLOCK_WIDTH,
                    self.BLOCK_WIDTH,
                ),
            )

        # Draw snakes
        self.snake1.draw(self.surface)
        self.snake2.draw(self.surface)

        # Draw scores
        score_text = self.font.render(
            f"Green: {self.snake1.score}  Blue: {self.snake2.score}",
            True,
            (255, 255, 255),
        )
        self.surface.blit(score_text, (10, 10))

        # Update display
        pygame.display.flip()

    def run(self, max_steps=10000):
        """Run the game"""
        step = 0

        while not self.game_over and step < max_steps:
            if self.display:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.game_over = True
                        break

            self.step()
            self.render()

            if self.display:
                self.clock.tick(10)  # 10 FPS
            
            # Progress update
            if not self.display and step % 500 == 0 and step > 0:
                print(f"Step {step}: Snake1 score={self.snake1.score}, Snake2 score={self.snake2.score}")

            step += 1

        if self.display:
            pygame.quit()
        
        # Show total apples spawned in this game
        total_apples = (step // self.apple_freq) + 1
        if not self.display:
            print(f"Game ended - Total apples spawned: {total_apples}, Snake1: {self.snake1.score}, Snake2: {self.snake2.score}")

        return self.snake1.score, self.snake2.score


def evaluate(
    model1_path,
    model2_path,
    apple_freq=100,
    freeze_time=30,
    max_steps=10000,
    num_rounds=100,
    display=True,
    state_size=16,
    action_size=4,
):
    """Evaluate two models in competitive play over multiple rounds"""
    total_scores = [0, 0]
    round_scores = []
    
    for round_num in range(num_rounds):
        if not display and round_num % 10 == 0:
            print(f"\nRound {round_num + 1}/{num_rounds}")
        
        game = CompetitiveGame(
            model1_path,
            model2_path,
            apple_freq=apple_freq,
            freeze_time=freeze_time,
            display=display,
            state_size=state_size,
            action_size=action_size,
        )
        scores = game.run(max_steps)
        
        total_scores[0] += scores[0]
        total_scores[1] += scores[1]
        round_scores.append(scores)
        
        if not display:
            print(f"Round {round_num + 1} scores - Model 1: {scores[0]}, Model 2: {scores[1]}")
    
    if not display:
        print(f"\n=== Final Results after {num_rounds} rounds ===")
        print(f"Total scores - Model 1: {total_scores[0]}, Model 2: {total_scores[1]}")
        print(f"Average scores - Model 1: {total_scores[0]/num_rounds:.2f}, Model 2: {total_scores[1]/num_rounds:.2f}")
        
        # Win statistics
        model1_wins = sum(1 for s in round_scores if s[0] > s[1])
        model2_wins = sum(1 for s in round_scores if s[1] > s[0])
        draws = sum(1 for s in round_scores if s[0] == s[1])
        print(f"Wins - Model 1: {model1_wins}, Model 2: {model2_wins}, Draws: {draws}")
    
    return total_scores, round_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate two snake models competitively"
    )
    parser.add_argument("model1", help="Path to first model")
    parser.add_argument("model2", help="Path to second model")
    parser.add_argument(
        "--apple_freq", type=int, default=100, help="Steps between apple spawns"
    )
    parser.add_argument(
        "--max_steps", type=int, default=10000, help="Maximum steps per game"
    )
    parser.add_argument(
        "--num_rounds", type=int, default=100, help="Number of rounds to play"
    )
    parser.add_argument("--no_display", action="store_true", help="Run without display")
    parser.add_argument(
        "--freeze_time", type=int, default=30, help="Steps to freeze when colliding"
    )
    parser.add_argument(
        "--state_size", type=int, default=16, help="Size of the state vector"
    )
    parser.add_argument(
        "--action_size", type=int, default=4, help="Number of possible actions"
    )

    args = parser.parse_args()

    total_scores, round_scores = evaluate(
        args.model1,
        args.model2,
        apple_freq=args.apple_freq,
        freeze_time=args.freeze_time,
        max_steps=args.max_steps,
        num_rounds=args.num_rounds,
        display=not args.no_display,
        state_size=args.state_size,
        action_size=args.action_size,
    )
