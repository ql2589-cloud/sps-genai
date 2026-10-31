from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from app.gan_model import Generator, Discriminator


BASE_DIR = Path(__file__).resolve().parent.parent


def main():
    torch.manual_seed(42)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Training device: {device}", flush=True)

    # Convert MNIST pixels to [-1, 1], matching Generator's Tanh.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(
        root=str(BASE_DIR / "data"),
        train=True,
        download=True,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0
    )

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()

    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=0.0002,
        betas=(0.5, 0.999)
    )
    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=0.0002,
        betas=(0.5, 0.999)
    )

    sample_dir = BASE_DIR / "gan_samples"
    sample_dir.mkdir(exist_ok=True)

    fixed_noise = torch.randn(64, 100, device=device)
    epochs = 20

    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        total_d = 0.0
        total_g = 0.0
        total_images = 0

        for step, (real_images, _) in enumerate(loader, start=1):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # Train D: real images -> 1, generated images -> 0.
            optimizer_d.zero_grad(set_to_none=True)

            noise = torch.randn(batch_size, 100, device=device)
            fake_images = generator(noise)

            loss_real = criterion(
                discriminator(real_images), real_labels
            )
            loss_fake = criterion(
                discriminator(fake_images.detach()), fake_labels
            )
            loss_d = loss_real + loss_fake

            loss_d.backward()
            optimizer_d.step()

            # Train G: make D classify generated images as real.
            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)

            optimizer_g.zero_grad(set_to_none=True)

            loss_g = criterion(
                discriminator(fake_images), real_labels
            )
            loss_g.backward()
            optimizer_g.step()

            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)

            total_d += loss_d.item() * batch_size
            total_g += loss_g.item() * batch_size
            total_images += batch_size

            if step % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Batch {step}/{len(loader)} | "
                    f"D: {loss_d.item():.4f} | "
                    f"G: {loss_g.item():.4f}",
                    flush=True
                )

        # Use the same noise each epoch to compare image quality.
        generator.eval()

        with torch.no_grad():
            samples = generator(fixed_noise).cpu()

        save_image(
            (samples + 1) / 2,
            sample_dir / f"epoch_{epoch + 1:02d}.png",
            nrow=8
        )

        torch.save(
            generator.cpu().state_dict(),
            BASE_DIR / "gan_generator.pth"
        )
        generator.to(device)

        print(
            f"Epoch {epoch + 1} completed | "
            f"Average D: {total_d / total_images:.4f} | "
            f"Average G: {total_g / total_images:.4f}",
            flush=True
        )

    print("Training finished. Saved gan_generator.pth", flush=True)


if __name__ == "__main__":
    main()