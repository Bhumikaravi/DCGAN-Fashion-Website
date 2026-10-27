import torch
from torch.utils.data import DataLoader, TensorDataset

from dcgan import Generator, Discriminator, initialize_weights
from train import train_one_epoch


def test_generator_output_shape_and_range():
    model = Generator()
    output = model(torch.randn(4, 64, 1, 1))
    assert output.shape == (4, 1, 28, 28)
    assert output.min().item() >= -1.0
    assert output.max().item() <= 1.0


def test_discriminator_output_shape():
    model = Discriminator()
    output = model(torch.randn(4, 1, 28, 28))
    assert output.shape == (4,)


def test_weights_are_finite():
    model = Generator()
    model.apply(initialize_weights)
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_training_loop_returns_finite_losses():
    generator = Generator()
    discriminator = Discriminator()
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)

    data = torch.randn(8, 1, 28, 28).clamp(-1, 1)
    labels = torch.zeros(8, dtype=torch.long)
    loader = DataLoader(TensorDataset(data, labels), batch_size=4)
    opt_g = torch.optim.Adam(generator.parameters(), lr=0.0002)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

    losses = train_one_epoch(
        generator, discriminator, loader, opt_g, opt_d,
        device="cpu", latent_dim=64
    )
    assert losses["generator_loss"] >= 0
    assert losses["discriminator_loss"] >= 0
    assert all(torch.isfinite(torch.tensor(v)).item() for v in losses.values())
