
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.cnn_model import CNNModel


def main():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = CNNModel().to(device)

    model.load_state_dict(
        torch.load(
            "cnn_model.pth",
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total

    print(f"Correct predictions: {correct}/{total}")
    print(f"Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
