import numpy as np
import math
import torch


class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = []

        self.visit_count = visit_count
        self.value_sum = 0

    def is_full_expanded(self):
        return len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return (
            q_value
            + self.args["C"] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
        )

    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, -1)

                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)

        return child

    def backpropagate(self, value):
        self.visit_count += 1
        self.value_sum += value

        value = self.game.get_opponent_value(value)

        if self.parent is not None:
            self.parent.backpropagate(value)


class MonteCarloTreeSearchParallel:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, states, self_play_games):
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
        )

        policy = torch.softmax(policy, axis=1).cpu().numpy()

        policy = (1 - self.args["dirichlet_epsilon"]) * policy + self.args[
            "dirichlet_epsilon"
        ] * np.random.dirichlet(
            [self.args["dirichlet_alpha"]] * self.game.action_size, size=policy.shape[0]
        )

        for i, self_play_game in enumerate(self_play_games):
            spg_policy = policy[i]
            valid_moves = self.game.get_valid_moves(states[i])
            spg_policy = spg_policy * valid_moves
            spg_policy = spg_policy / np.sum(spg_policy)

            self_play_game.root = Node(self.game, self.args, states[i], visit_count=1)
            self_play_game.root.expand(spg_policy)

        for search in range(self.args["num_searches"]):
            for self_play_game in self_play_games:
                self_play_game.node = None
                node = self_play_game.root

                while node.is_full_expanded():
                    node = node.select()

                value, is_terminal = self.game.get_value_and_terminated(
                    node.state, node.action_taken
                )
                value = self.game.get_opponent_value(value)

                if is_terminal:
                    node.backpropagate(value)
                else:
                    self_play_game.node = node

            expandable_self_play_games = [
                mapping_idx
                for mapping_idx in range(len(self_play_games))
                if self_play_games[mapping_idx].node is not None
            ]

            if len(expandable_self_play_games) > 0:
                states = np.stack(
                    [
                        self_play_games[mapping_idx].node.state
                        for mapping_idx in expandable_self_play_games
                    ]
                )
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
                )
                policy = torch.softmax(policy, axis=1).cpu().numpy()

                for i, mapping_idx in enumerate(expandable_self_play_games):
                    node = self_play_games[mapping_idx].node
                    spg_policy, spg_value = policy[i], value[i]

                    valid_moves = self.game.get_valid_moves(node.state)
                    spg_policy = spg_policy * valid_moves
                    spg_policy = spg_policy / np.sum(spg_policy)

                    node.expand(spg_policy)
                    node.backpropagate(spg_value)


class SPG:
    # Self Play Game
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None
