# DCGAN Fashion Studio

A small Streamlit website that trains a Deep Convolutional Generative Adversarial Network (DCGAN) to generate 28x28 grayscale clothing-like images.

## Features
- Streamlit website with dataset, epoch, batch-size, and learning-rate controls
- Real Fashion-MNIST dataset support with a small subset option
- Synthetic clothing-like images for testing the pipeline
- PyTorch Generator and Discriminator
- Actual adversarial training loop
- Epoch progress and generator/discriminator loss records
- Generated image gallery
- Checkpoint saving and loading
- Downloadable CSV loss history
- Pytest test cases

## Run locally

Install Python 3.10 or 3.11 if possible. In this folder, open a terminal and run:

```bash
python -m venv .venv
```

Windows:
```bash
.venv\\Scripts\\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Launch the website:
```bash
streamlit run app.py
```

Run tests:
```bash
pytest -v
```

The app generates synthetic images automatically when Synthetic is selected. Choose Fashion-MNIST to download and train on real clothing images; internet access is needed for the first download.

## Deploy to Streamlit Community Cloud
1. Push the project folder to a GitHub repository.
2. Open Streamlit Community Cloud and create a new app.
3. Select your repository and set the main file to `app.py`.
4. Deploy.

Cloud resources are limited, so use a small dataset and few batches per epoch. Local checkpoints may not persist across cloud restarts.

## Architecture
- Generator: transposed convolutions, BatchNorm, ReLU, Tanh
- Discriminator: convolutions, BatchNorm, LeakyReLU, logits
- Loss: BCEWithLogitsLoss
- Optimizer: Adam
- Latent dimension: 64

## Important
Synthetic shapes are not real Fashion-MNIST samples. They help verify that the app and training pipeline run. Evaluate generated quality using the gallery and training records; GAN losses do not need to decrease steadily.
