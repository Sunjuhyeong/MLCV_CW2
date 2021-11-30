import torch
import torch.nn as nn


class CNNClassifier(nn.Module):

    def __init__(self):
        super(CNNClassifier, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4)
        )

        self.linear = nn.Sequential(
            nn.Linear(4 * 4 * 256, 1024, bias=True),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            nn.Linear(1024, 256, bias=True),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 10)
        )

        self.apply(self.init_weights)

        print(f"number of total parameters for G: {sum(p.numel() for p in self.parameters())}")

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            m.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)
        out = self.linear(out.view(x.size(0), -1))
        return out
