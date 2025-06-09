import json
import os
import random
from collections import deque

import torch.types
from torch import nn, optim
import torch.nn.functional as F
import numpy as np

# from game import Game
from game_no_ui import Game


class ANN(nn.Module):
    def __init__(self, state_size, action_size):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        return self.fc2(x)


class ReplayMemory:
    def __init__(self, capacity):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        # event - (state, action, reward, next_state, done)
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, k):
        experiences = random.sample(self.memory, k=k)
        # [(state, action, reward, next_state, done)]
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        return states, actions, rewards, next_states, dones


# Hyperparameters
number_episodes = 100000  # Defines how many episodes the agent will train on.
maximum_number_steps_per_episode = 200000  # Sets the limit on how long an episode can last.
epsilon_starting_value = 1.0  # Initial value for the exploration-exploitation tradeoff.
epsilon_ending_value = 0.001  # Final value for epsilon after decay.
epsilon_decay_value = 0.99  # Rate at which epsilon decreases over time.
learning_rate = 0.01  # The step size used by the optimizer to update weights.
minibatch_size = 100  # Number of samples used in each training step.
gamma = 0.95  # Discount factor for future rewards in Q-learning.
replay_buffer_size = int(1e5)  # Maximum capacity of the replay memory.
interpolation_parameter = 1e-2  # Parameter for soft updates in the target network.

state_size = 16
action_size = 4
scores_on_100_episodes = deque(maxlen=100)
folder = "model"


class Agent:
    def __init__(self, state_size, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.local_network = ANN(state_size, action_size).to(self.device)
        self.target_network = ANN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_network.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(replay_buffer_size)
        self.t_step = 0
        self.record = -1
        self.epsilon = -1

    def get_state(self, game):
        head_x = game.snake.x[0]
        head_y = game.snake.y[0]

        point_left = [(head_x - game.BLOCK_WIDTH), head_y]
        point_right = [(head_x + game.BLOCK_WIDTH), head_y]
        point_up = [head_x, (head_y - game.BLOCK_WIDTH)]
        point_down = [head_x, (head_y + game.BLOCK_WIDTH)]
        point_left_up = [(head_x - game.BLOCK_WIDTH), (head_y - game.BLOCK_WIDTH)]
        point_left_down = [(head_x - game.BLOCK_WIDTH), (head_y + game.BLOCK_WIDTH)]
        point_right_up = [(head_x + game.BLOCK_WIDTH), (head_y - game.BLOCK_WIDTH)]
        point_right_down = [(head_x + game.BLOCK_WIDTH), (head_y + game.BLOCK_WIDTH)]

        # [danger right, danger down, danger left, danger up, danger right-up,
        # danger right-down, danger left-down, danger left-up,
        # direction left, direction right, direction up, direction down,
        # food left, food right, food up, food down]

        state = [
            game.is_danger(point_left),
            game.is_danger(point_right),
            game.is_danger(point_up),
            game.is_danger(point_down),
            game.is_danger(point_left_up),
            game.is_danger(point_left_down),
            game.is_danger(point_right_up),
            game.is_danger(point_right_down),

            # move direction
            game.snake.direction == "left",
            game.snake.direction == "right",
            game.snake.direction == "up",
            game.snake.direction == "down",

            # food location - compare food location with snake head
            game.apple.x < head_x,  # food left
            game.apple.x > head_x,  # food right
            game.apple.y < head_y,  # food up
            game.apple.y < head_y,  # food down
        ]

        return np.array(state, dtype=int)

    def step(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            if len(self.memory.memory) > minibatch_size:
                experiences = self.memory.sample(k=minibatch_size)
                self.learn(experiences)

    def get_action(self, state, epsilon):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_network.eval()
        with torch.no_grad():
            action_values = self.local_network(state)  # [Q(state, action_0), Q(state, action_1) .....]
        self.local_network.train()
        if random.random() > epsilon:
            move = torch.argmax(action_values).item()
        else:
            move = random.randint(0, 3)

        return move

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        next_q_targets = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + gamma * next_q_targets * (1 - dones)
        q_expected = self.local_network(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.local_network, self.target_network)

    def soft_update(self, local_network, target_network):
        for local_params, target_params in zip(local_network.parameters(), target_network.parameters()):
            target_params.data.copy_(
                interpolation_parameter * local_params + (1.0 - interpolation_parameter) * target_params
            )

    def load(self, file_name='model.pth'):
        file_path = os.path.join(folder, file_name)
        if os.path.exists(file_path):
            self.local_network.load_state_dict(torch.load(file_path))
            print("Model Loaded")
            self.retrieve_data()

    def save_model(self, file_name='model.pth'):
        if not os.path.exists(folder):
            os.mkdir(folder)

        file_name = os.path.join(folder, file_name)
        torch.save(self.local_network.state_dict(), file_name)

    def retrieve_data(self):
        file_name = "data.json"
        model_data_path = os.path.join(folder, file_name)
        if os.path.exists(model_data_path):
            with open(model_data_path, 'r') as file:
                data = json.load(file)

                if data is not None:
                    self.record = data['record']
                    self.epsilon = data['epsilon']

    def save_data(self, record, epsilon):
        file_name = "data.json"
        if not os.path.exists(folder):
            os.mkdir(folder)

        complete_path = os.path.join(folder, file_name)
        data = {'record': record, 'epsilon': epsilon}
        with open(complete_path, 'w') as file:
            json.dump(data, file, indent=4)


if __name__ == "__main__":
    game = Game()
    agent = Agent(state_size=state_size, action_size=action_size)
    agent.load()
    max_score = 0

    epsilon = epsilon_starting_value

    if agent.epsilon != -1:
        epsilon = agent.epsilon
        max_score = max(agent.record, max_score)
    print('epsilon starts at {}', epsilon)
    for episode in range(0, number_episodes):
        game.reset()
        score = 0

        for t in range(maximum_number_steps_per_episode):
            state_old = agent.get_state(game)
            action = agent.get_action(state_old, epsilon)
            # perform the action
            move = [0, 0, 0, 0]
            move[action] = 1
            reward, done, score = game.run(move)
            state_new = agent.get_state(game)
            agent.step(state_old, action, reward, state_new, done)
            if done:
                break
        max_score = max(max_score, score)
        scores_on_100_episodes.append(score)
        epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)
        agent.save_model()
        agent.save_data(max_score, epsilon)
        if episode % 50 == 0:
            print('Episode {}\t Curr Score {}\tMax Score {}\tAvg Score {:.2f}'.format(episode, score, max_score,
                                                                                      np.mean(scores_on_100_episodes)))
