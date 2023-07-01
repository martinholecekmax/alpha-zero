from ticktacktoe import TicTacToe
from connect_four import ConnectFour
from resnet import ResNet
import torch
from alphazero import AlphaZero

# game = TicTacToe()
game = ConnectFour()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(game, 9, 128, device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

args_tic = {
    "C": 2,
    "num_searches": 60,
    "num_iterations": 3,
    "num_self_plays": 500,
    "num_epochs": 4,
    "batch_size": 64,
    "temperature": 1.25,
    "dirichlet_epsilon": 0.25,
    "dirichlet_alpha": 0.3,
}

args = {
    "C": 2,
    "num_searches": 600,
    "num_iterations": 8,
    "num_self_plays": 500,
    "num_epochs": 4,
    "batch_size": 128,
    "temperature": 1.25,
    "dirichlet_epsilon": 0.25,
    "dirichlet_alpha": 0.3,
}

alphazero = AlphaZero(model, optimizer, game, args)
alphazero.learn()
