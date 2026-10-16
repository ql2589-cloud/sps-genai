
import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Part 1: Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Part 2: Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = CNNModel()

    # A batch containing one random RGB image
    sample = torch.randn(1, 3, 64, 64)

    output = model(sample)

    print("Input shape:", sample.shape)
    print("Output shape:", output.shape)
