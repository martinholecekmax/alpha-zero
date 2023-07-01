import torch
import torch.nn as nn
import torch.nn.functional as F

print("torch version: ", torch.__version__)


class ResNet(nn.Module):
    def __init__(self, game, num_res_blocks, num_hidden, device):
        super().__init__()
        self.device = device
        self.game = game
        self.num_res_blocks = num_res_blocks
        self.num_hidden = num_hidden
        self.start_block = nn.Sequential(
            nn.Conv2d(3, self.num_hidden, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_hidden),
            nn.ReLU(),
        )
        self.backbone = nn.ModuleList(
            [ResBlock(self.num_hidden) for _ in range(self.num_res_blocks)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(self.num_hidden, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * self.game.row_count * self.game.col_count, self.game.action_size),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(self.num_hidden, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * self.game.row_count * self.game.col_count, 1),
            nn.Tanh(),
        )

        self.to(self.device)

    def forward(self, x):
        x = self.start_block(x)
        for res_block in self.backbone:
            x = res_block(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.num_hidden = num_hidden
        self.conv1 = nn.Conv2d(self.num_hidden, self.num_hidden, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_hidden)
        self.conv2 = nn.Conv2d(self.num_hidden, self.num_hidden, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_hidden)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += residual
        x = F.relu(x)
        return x
