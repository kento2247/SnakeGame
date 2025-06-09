import json
import os
import random
from collections import deque

import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
from src.model import ANN
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Snake Game DQN Training")

    parser.add_argument(
        "--number_episodes",
        type=int,
        default=100000,
        help="Defines how many episodes the agent will train on",
    )
    parser.add_argument(
        "--maximum_number_steps_per_episode",
        type=int,
        default=200000,
        help="Sets the limit on how long an episode can last",
    )
    parser.add_argument(
        "--epsilon_starting_value",
        type=float,
        default=1.0,
        help="Initial value for the exploration-exploitation tradeoff",
    )
    parser.add_argument(
        "--epsilon_ending_value",
        type=float,
        default=0.001,
        help="Final value for epsilon after decay",
    )
    parser.add_argument(
        "--epsilon_decay_value",
        type=float,
        default=0.99,
        help="Rate at which epsilon decreases over time",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="The step size used by the optimizer to update weights",
    )
    parser.add_argument(
        "--minibatch_size",
        type=int,
        default=100,
        help="Number of samples used in each training step",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Discount factor for future rewards in Q-learning",
    )
    parser.add_argument(
        "--replay_buffer_size",
        type=int,
        default=int(1e5),
        help="Maximum capacity of the replay memory",
    )
    parser.add_argument(
        "--interpolation_parameter",
        type=float,
        default=1e-2,
        help="Parameter for soft updates in the target network",
    )
    parser.add_argument("--state_size", type=int, default=16)
    parser.add_argument("--action_size", type=int, default=4)
    parser.add_argument(
        "--folder", type=str, default="model", help="Folder to save model checkpoints"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run the game with a graphical user interface",
    )

    args = parser.parse_args()
    args.scores_on_100_episodes = deque(maxlen=100)

    return args


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
        states = torch.tensor(
            np.vstack([e[0] for e in experiences if e is not None]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.tensor(
            np.vstack([e[1] for e in experiences if e is not None]),
            dtype=torch.long,
            device=self.device,
        )
        rewards = torch.tensor(
            np.vstack([e[2] for e in experiences if e is not None]),
            dtype=torch.float32,
            device=self.device,
        )
        next_states = torch.tensor(
            np.vstack([e[3] for e in experiences if e is not None]),
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.tensor(
            np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8),
            dtype=torch.float32,
            device=self.device,
        )

        return states, actions, rewards, next_states, dones


class Agent:
    def __init__(self, state_size, action_size, args):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.local_network = ANN(state_size, action_size).to(self.device)
        self.target_network = ANN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(
            self.local_network.parameters(), lr=args.learning_rate
        )
        self.memory = ReplayMemory(args.replay_buffer_size)
        self.t_step = 0
        self.record = -1
        self.epsilon = -1
        self.args = args

        print("Using device: {}".format("cuda" if torch.cuda.is_available() else "cpu"))

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

        state = [
            game.is_danger(point_left),
            game.is_danger(point_right),
            game.is_danger(point_up),
            game.is_danger(point_down),
            game.is_danger(point_left_up),
            game.is_danger(point_left_down),
            game.is_danger(point_right_up),
            game.is_danger(point_right_down),
            game.snake.direction == "left",
            game.snake.direction == "right",
            game.snake.direction == "up",
            game.snake.direction == "down",
            game.apple.x < head_x,
            game.apple.x > head_x,
            game.apple.y < head_y,
            game.apple.y > head_y,
        ]

        return torch.tensor(state, dtype=torch.int32)

    def step(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            if len(self.memory.memory) > self.args.minibatch_size:
                experiences = self.memory.sample(k=self.args.minibatch_size)
                self.learn(experiences)

    def get_action(self, state, epsilon):
        state = state.float().unsqueeze(0).to(self.device)
        self.local_network.eval()
        with torch.no_grad():
            action_values = self.local_network(state)
        self.local_network.train()
        if random.random() > epsilon:
            move = torch.argmax(action_values).item()
        else:
            move = random.randint(0, 3)

        return move

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        next_q_targets = (
            self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        )
        q_targets = rewards + self.args.gamma * next_q_targets * (1 - dones)
        q_expected = self.local_network(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.local_network, self.target_network)

    def soft_update(self, local_network, target_network):
        for local_params, target_params in zip(
            local_network.parameters(), target_network.parameters()
        ):
            target_params.data.copy_(
                self.args.interpolation_parameter * local_params
                + (1.0 - self.args.interpolation_parameter) * target_params
            )

    def load(self, file_name="model.pth"):
        file_path = os.path.join(self.args.folder, file_name)
        if os.path.exists(file_path):
            self.local_network.load_state_dict(torch.load(file_path))
            print("Model Loaded")
            self.retrieve_data()

    def save_model(self, file_name="model.pth"):
        if not os.path.exists(self.args.folder):
            os.mkdir(self.args.folder)

        file_name = os.path.join(self.args.folder, file_name)
        torch.save(self.local_network.state_dict(), file_name)

    def retrieve_data(self):
        file_name = "data.json"
        model_data_path = os.path.join(self.args.folder, file_name)
        if os.path.exists(model_data_path):
            with open(model_data_path, "r") as file:
                data = json.load(file)

                if data is not None:
                    self.record = data["record"]
                    self.epsilon = data["epsilon"]

    def save_data(self, record, epsilon):
        file_name = "data.json"
        if not os.path.exists(self.args.folder):
            os.mkdir(self.args.folder)

        complete_path = os.path.join(self.args.folder, file_name)
        data = {"record": record, "epsilon": epsilon}
        with open(complete_path, "w") as file:
            json.dump(data, file, indent=4)


if __name__ == "__main__":
    args = get_args()
    if args.gui:
        from src.game import Game
    else:
        from src.game_no_ui import Game
    game = Game()

    agent = Agent(state_size=args.state_size, action_size=args.action_size, args=args)
    agent.load()
    max_score = 0

    epsilon = args.epsilon_starting_value

    if agent.epsilon != -1:
        args.epsilon_starting_value = agent.epsilon
        max_score = max(agent.record, max_score)
    print("epsilon starts at {}".format(args.epsilon_starting_value))
    args.scores_on_100_episodes = deque(maxlen=100)

    for episode in range(0, args.number_episodes):
        game.reset()
        score = 0

        for t in range(args.maximum_number_steps_per_episode):
            state_old = agent.get_state(game)
            action = agent.get_action(state_old, args.epsilon_starting_value)
            # perform the action
            move = [0, 0, 0, 0]
            move[action] = 1
            reward, done, score = game.run(move)
            state_new = agent.get_state(game)
            agent.step(state_old, action, reward, state_new, done)
            if done:
                break
        max_score = max(max_score, score)
        args.scores_on_100_episodes.append(score)
        args.epsilon_starting_value = max(
            args.epsilon_ending_value,
            args.epsilon_decay_value * args.epsilon_starting_value,
        )
        agent.save_model()
        agent.save_data(max_score, args.epsilon_starting_value)
        if episode % 50 == 0:
            print(
                "Episode {}\t Curr Score {}\tMax Score {}\tAvg Score {:.2f}".format(
                    episode, score, max_score, np.mean(args.scores_on_100_episodes)
                )
            )
