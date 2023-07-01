import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random
from monte_carlo_tree_search_parallel import SPG, MonteCarloTreeSearchParallel


class AlphaZeroParallel:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args

        self.mcts = MonteCarloTreeSearchParallel(game, args, model)

    def self_play(self):
        return_memory = []
        player = 1
        self_play_games = [SPG(self.game) for _ in range(self.args["num_parallel_games"])]

        while len(self_play_games) > 0:
            states = np.stack([spg.state for spg in self_play_games])

            neutral_states = self.game.change_perspective(states, player)

            self.mcts.search(neutral_states, self_play_games)

            for i in range(len(self_play_games))[::-1]:
                self_play_game = self_play_games[i]

                action_probs = np.zeros(self.game.action_size)
                for child in self_play_game.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs = action_probs / np.sum(action_probs)

                self_play_game.memory.append((self_play_game.root.state, action_probs, player))

                temperature_action_probs = action_probs ** (1 / self.args["temperature"])
                temperature_action_probs /= np.sum(temperature_action_probs)
                action = np.random.choice(self.game.action_size, p=temperature_action_probs)

                self_play_game.state = self.game.get_next_state(
                    self_play_game.state, action, player
                )
                value, is_terminal = self.game.get_value_and_terminated(
                    self_play_game.state, action
                )

                if is_terminal:
                    for hist_neutral_state, hist_action_probs, hist_player in self_play_game.memory:
                        hist_outcome = (
                            value if hist_player == player else self.game.get_opponent(value)
                        )
                        return_memory.append(
                            (
                                self.game.get_encoded_state(hist_neutral_state),
                                hist_action_probs,
                                hist_outcome,
                            )
                        )
                    del self_play_games[i]

            player = self.game.get_opponent(player)

        return return_memory

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
            for self_play_iteration in tqdm(
                range(self.args["num_self_plays"] // self.args["num_parallel_games"])
            ):
                memory += self.self_play()

            self.model.train()
            for epoch in tqdm(range(self.args["num_epochs"])):
                self.train(memory)

            torch.save(self.model.state_dict(), f"weights/model_{iteration}_{self.game}.pt")
            torch.save(self.optimizer.state_dict(), f"weights/optimizer_{iteration}_{self.game}.pt")
