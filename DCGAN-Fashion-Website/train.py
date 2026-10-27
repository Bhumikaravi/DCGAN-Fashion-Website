from pathlib import Path
import torch
from torch import nn


def train_one_epoch(generator, discriminator, loader, optimizer_g, optimizer_d,
                    device="cpu", latent_dim=64, max_batches=None):
    generator.train()
    discriminator.train()
    criterion = nn.BCEWithLogitsLoss()
    total_g, total_d, batches = 0.0, 0.0, 0

    for batch_index, (real, _) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        real = real.to(device)
        size = real.size(0)
        real_labels = torch.ones(size, device=device)
        fake_labels = torch.zeros(size, device=device)

        # Discriminator update: classify real and generated images.
        optimizer_d.zero_grad(set_to_none=True)
        real_logits = discriminator(real).view(-1)
        loss_real = criterion(real_logits, real_labels)

        noise = torch.randn(size, latent_dim, 1, 1, device=device)
        fake = generator(noise)
        fake_logits = discriminator(fake.detach()).view(-1)
        loss_fake = criterion(fake_logits, fake_labels)

        d_loss = loss_real + loss_fake
        d_loss.backward()
        optimizer_d.step()

        # Generator update: try to make fake images classify as real.
        optimizer_g.zero_grad(set_to_none=True)
        generated_logits = discriminator(fake).view(-1)
        g_loss = criterion(generated_logits, real_labels)
        g_loss.backward()
        optimizer_g.step()

        total_g += float(g_loss.item())
        total_d += float(d_loss.item())
        batches += 1

    if batches == 0:
        raise ValueError("No batches processed. Check dataset and batch size.")

    return {
        "generator_loss": total_g / batches,
        "discriminator_loss": total_d / batches,
    }


def save_checkpoint(path, generator, discriminator, optimizer_g, optimizer_d,
                    epoch, history, latent_dim=64):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": int(epoch),
        "latent_dim": int(latent_dim),
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "history": history,
    }, path)


def load_checkpoint(path, generator, discriminator, optimizer_g, optimizer_d,
                    device="cpu"):
    # Load checkpoints that you created or otherwise trust.
    try:
        checkpoint = torch.load(
            path, map_location=device, weights_only=True
        )
    except TypeError:  # Compatibility with older PyTorch versions.
        checkpoint = torch.load(path, map_location=device)

    generator.load_state_dict(checkpoint["generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])
    optimizer_g.load_state_dict(checkpoint["optimizer_g"])
    optimizer_d.load_state_dict(checkpoint["optimizer_d"])
    return checkpoint["epoch"], checkpoint["history"]
