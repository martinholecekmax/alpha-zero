import kaggle_environments
from monte_carlo_tree_search_parallel import MonteCarloTreeSearchParallel
import numpy as np
from connect_four import ConnectFour
from resnet import ResNet
import torch

print(kaggle_environments.__version__)


class KaggleAgent:
    def __init__(self, model, game, args):
        self.model = model
        self.game = game
        self.args = args
        if self.args["search"]:
            self.mcts = MonteCarloTreeSearchParallel(self.game, self.args, self.model)

    def run(self, obs, conf):
        player = obs["mark"] if obs["mark"] == 1 else -1
        state = np.array(obs["board"]).reshape(self.game.row_count, self.game.col_count)
        state[state == 2] = -1

        state = self.game.change_perspective(state, player)

        if self.args["search"]:
            policy = self.mcts.search(state)
        else:
            policy, _ = self.model.predict(state, augment=self.args["augment"])

        valid_moves = self.game.get_valid_moves(state)
        policy = policy * valid_moves
        policy = policy / np.sum(policy)

        if self.args["temperature"] == 0:
            action = int(np.argmax(policy))
        elif self.args["temperature"] == float("inf"):
            action = int(
                np.random.choice([r for r in range(self.game.action_size) if policy[r] > 0])
            )
        else:
            policy = policy ** (1 / self.args["temperature"])
            policy /= np.sum(policy)
            action = int(np.random.choice(self.game.action_size, p=policy))

        return action


game = ConnectFour()

args = {
    "C": 2,
    "num_searches": 600,
    "dirichlet_epsilon": 0.1,
    "dirichlet_alpha": 0.3,
    "temperature": 0,
    "search": True,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(game, 9, 128, device=device)
model.load_state_dict(torch.load("weights/model_7_ConnectFour.pt", map_location=device))
model.eval()

env = kaggle_environments.make("connectx")

player1 = KaggleAgent(model, game, args)
player2 = KaggleAgent(model, game, args)

players = [player1.run, player2.run]

env.run(players)
env.render(mode="ipython", width=500, height=450)
