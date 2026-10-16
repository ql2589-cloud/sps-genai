
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.cnn_model import CNNModel


def main():
    # 1. Convert images to 64x64 tensors
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # 2. Load CIFAR-10 training dataset
    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )

    # 3. Create CNN model
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = CNNModel().to(device)

    # 4. Loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # 5. Training loop
    epochs = 5

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {average_loss:.4f}"
        )

    # 6. Save trained weights
    torch.save(
        model.state_dict(),
        "cnn_model.pth"
    )

    print("Training completed!")
    print("Model saved as cnn_model.pth")


if __name__ == "__main__":
    main()
