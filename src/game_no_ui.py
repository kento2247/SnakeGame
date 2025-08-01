import random

from sympy.physics.control.control_plots import np

from src.game_base import GameBase


class Snake(GameBase):
    def __init__(self, length=5):
        super().__init__()
        self.length = length
        # Initialize snake at a random position
        start_x = random.randint(1, self.MAX_FOOD_INDEX - 1) * self.BLOCK_WIDTH
        start_y = random.randint(1, self.MAX_FOOD_INDEX - 1) * self.BLOCK_WIDTH
        self.x = [start_x] * self.length
        self.y = [start_y] * self.length
        self.direction = "right"

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
    def __init__(self):
        super().__init__()
        self.x = random.randint(0, self.MAX_FOOD_INDEX) * self.BLOCK_WIDTH
        self.y = random.randint(0, self.MAX_FOOD_INDEX) * self.BLOCK_WIDTH

    def move(self, snake):
        while True:  # make sure new food is not getting created over snake body
            x = random.randint(0, self.MAX_FOOD_INDEX) * self.BLOCK_WIDTH
            y = random.randint(0, self.MAX_FOOD_INDEX) * self.BLOCK_WIDTH
            clean = True
            for i in range(0, snake.length):
                if x == snake.x[i] and y == snake.y[i]:
                    clean = False
                    break
            if clean:
                self.x = x
                self.y = y
                return


class Game(GameBase):
    def __init__(self):
        super().__init__()
        self.snake = Snake(length=1)
        self.apple = Apple()
        self.score = 0
        self.game_over = False
        self.reward = 0

    def play(self):
        self.snake.move()
        self.reward = self.reward_function()

    def is_collision(self):
        head_x = self.snake.x[0]
        head_y = self.snake.y[0]

        for i in range(1, self.snake.length):
            if head_x == self.snake.x[i] and head_y == self.snake.y[i]:
                return True

        if (
            head_x > (self.SCREEN_SIZE - self.BLOCK_WIDTH)
            or head_y > (self.SCREEN_SIZE - self.BLOCK_WIDTH)
            or head_x < 0
            or head_y < 0
        ):
            return True

        return False

    def is_danger(self, point):
        point_x = point[0]
        point_y = point[1]

        for i in range(1, self.snake.length):
            if point_x == self.snake.x[i] and point_y == self.snake.y[i]:
                return True

        if (
            point_x > (self.SCREEN_SIZE - self.BLOCK_WIDTH)
            or point_y > (self.SCREEN_SIZE - self.BLOCK_WIDTH)
            or point_x < 0
            or point_y < 0
        ):
            return True

        return False

    def reset(self):
        self.snake = Snake()
        self.apple = Apple()
        self.score = 0
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

    def run(self, move):
        dir = self.get_next_direction(move)

        if dir == "left":
            self.snake.move_left()
        elif dir == "right":
            self.snake.move_right()
        elif dir == "down":
            self.snake.move_down()
        elif dir == "up":
            self.snake.move_up()
        self.play()

        return self.reward, self.game_over, self.score
