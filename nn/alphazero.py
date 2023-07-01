from monte_carlo_tree_search import MonteCarloTreeSearch
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random


class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args

        self.mcts = MonteCarloTreeSearch(game, args, model)

    def self_play(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()

        while True:
            neutral_state = self.game.change_perspective(state, player)
            action_probs = self.mcts.search(neutral_state)

            memory.append((neutral_state, action_probs, player))

            temperature_action_probs = action_probs ** (1 / self.args["temperature"])
            temperature_action_probs /= np.sum(temperature_action_probs)
            action = np.random.choice(self.game.action_size, p=temperature_action_probs)

            state = self.game.get_next_state(state, action, player)
            value, is_terminal = self.game.get_value_and_terminated(state, action)

            if is_terminal:
                memory_out = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent(value)
                    memory_out.append(
                        (
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome,
                        )
                    )

                return memory_out

            player = self.game.get_opponent(player)

    def train(self, memory):
        random.shuffle(memory)
        for batch_idx in range(0, len(memory), self.args["batch_size"]):
            sample = memory[batch_idx : min(len(memory) - 1, batch_idx + self.args["batch_size"])]

            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = (
                np.array(state),
                np.array(policy_targets),
                np.array(value_targets).reshape(-1, 1),
            )

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(
                policy_targets, dtype=torch.float32, device=self.model.device
            )
            value_targets = torch.tensor(
                value_targets, dtype=torch.float32, device=self.model.device
            )

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)

            loss = value_loss + policy_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        for iteration in range(self.args["num_iterations"]):
            memory = []

            self.model.eval()
            for self_play_iteration in tqdm(range(self.args["num_self_plays"])):
                memory += self.self_play()

            self.model.train()
            for epoch in tqdm(range(self.args["num_epochs"])):
                self.train(memory)

            torch.save(self.model.state_dict(), f"weights/model_{iteration}_{self.game}.pt")
            torch.save(self.optimizer.state_dict(), f"weights/optimizer_{iteration}_{self.game}.pt")
