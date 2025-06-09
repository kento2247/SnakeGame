class GameBase:
    def __init__(self):
        self.SCREEN_SIZE = 600
        self.BLOCK_WIDTH = 20
        self.MAX_FOOD_INDEX = (self.SCREEN_SIZE - self.BLOCK_WIDTH) // self.BLOCK_WIDTH

    def reward_function(self):
        current_distance = abs(self.snake.x[0] - self.apple.x) + abs(
            self.snake.y[0] - self.apple.y
        )
        # Base reward for each step
        reward = -0.1
        # Additional reward based on distance to apple
        if current_distance < self.BLOCK_WIDTH * 2:  # Very close to apple
            return reward + 0.5
        elif current_distance < self.BLOCK_WIDTH * 4:  # Moderately close to apple
            return reward + 0.2
        # if snake eats the apple
        if self.snake.x[0] == self.apple.x and self.snake.y[0] == self.apple.y:
            self.score += 1
            self.snake.increase()
            self.apple.move(self.snake)
            reward = 10
        if self.is_collision():
            self.game_over = True
            reward = -100
        return reward
