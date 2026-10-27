
from pathlib import Path
import io

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch
from torchvision.utils import make_grid

from dcgan import (
    Generator,
    Discriminator,
    initialize_weights,
    LATENT_DIM,
)
from dataset import get_dataloader
from train import (
    train_one_epoch,
    save_checkpoint,
    load_checkpoint,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DCGAN Fashion Studio",
    page_icon="👕",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = CHECKPOINT_DIR / "dcgan_checkpoint.pth"

DATA_LIMIT = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #faf7ff, #f2edff);
    }
    .hero {
        background: linear-gradient(120deg, #6d28d9, #a78bfa);
        color: white;
        padding: 28px;
        border-radius: 18px;
        margin-bottom: 20px;
    }
    .hero h1 {
        color: white;
        margin-bottom: 8px;
    }
    .small-note {
        color: #6b5b85;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. MODEL HELPERS
# ============================================================

def create_models(learning_rate):
    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=learning_rate,
        betas=(0.5, 0.999),
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=learning_rate,
        betas=(0.5, 0.999),
    )

    return generator, discriminator, optimizer_g, optimizer_d


def initialize_session(learning_rate):
    (
        st.session_state.generator,
        st.session_state.discriminator,
        st.session_state.optimizer_g,
        st.session_state.optimizer_d,
    ) = create_models(learning_rate)

    st.session_state.epoch_history = []
    st.session_state.generated_images = None
    st.session_state.models_ready = True
    st.session_state.training_complete = False


if "models_ready" not in st.session_state:
    st.session_state.models_ready = False

if "epoch_history" not in st.session_state:
    st.session_state.epoch_history = []

if "generated_images" not in st.session_state:
    st.session_state.generated_images = None

if "training_complete" not in st.session_state:
    st.session_state.training_complete = False


# ============================================================
# 4. HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>👕 DCGAN Fashion Studio</h1>
        Train a Deep Convolutional Generative Adversarial Network
        and generate new fashion-like images using AI.
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    f"Device: {str(DEVICE).upper()} · "
    f"Training image limit: {DATA_LIMIT}"
)


# ============================================================
# 5. SIDEBAR SETTINGS
# ============================================================

with st.sidebar:
    st.header("⚙️ Training Controls")

    dataset_source = st.selectbox(
        "Choose dataset",
        ["Fashion-MNIST", "Synthetic"],
        key="dataset_selector",
    )

    epochs = st.slider(
        "Number of epochs",
        min_value=1,
        max_value=20,
        value=2,
    )

    batch_size = st.selectbox(
        "Batch size",
        [8, 16, 32, 64],
        index=1,
    )

    learning_rate = st.selectbox(
        "Learning rate",
        [0.0001, 0.0002, 0.0005, 0.001],
        index=1,
        format_func=lambda value: f"{value:g}",
    )

    max_batches = st.number_input(
        "Maximum batches per epoch (0 = all)",
        min_value=0,
        max_value=100,
        value=0,
        step=1,
    )

    st.info(
        "Fashion-MNIST is downloaded automatically when possible. "
        "Synthetic mode requires image files to exist in the project."
    )

    start_button = st.button(
        "🚀 Start Training",
        use_container_width=True,
        type="primary",
    )

    generate_button = st.button(
        "🖼️ Generate Images",
        use_container_width=True,
    )

    save_button = st.button(
        "💾 Save Checkpoint",
        use_container_width=True,
    )

    load_button = st.button(
        "📂 Load Checkpoint",
        use_container_width=True,
    )


# ============================================================
# 6. MODEL OVERVIEW
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset", dataset_source)

with col2:
    st.metric("Image limit", DATA_LIMIT)

with col3:
    st.metric("Compute device", str(DEVICE).upper())


# ============================================================
# 7. LOAD CHECKPOINT
# ============================================================

if load_button:
    if CHECKPOINT_PATH.exists():
        try:
            if not st.session_state.models_ready:
                initialize_session(learning_rate)

            saved_epoch, saved_history = load_checkpoint(
                str(CHECKPOINT_PATH),
                st.session_state.generator,
                st.session_state.discriminator,
                st.session_state.optimizer_g,
                st.session_state.optimizer_d,
                device=str(DEVICE),
            )

            st.session_state.epoch_history = saved_history
            st.session_state.training_complete = True

            st.success(
                f"Checkpoint loaded successfully. "
                f"Saved epoch: {saved_epoch}"
            )

        except Exception as exc:
            st.error("Could not load the checkpoint.")
            st.exception(exc)
    else:
        st.warning(
            f"No checkpoint found at {CHECKPOINT_PATH}. "
            "Train the model and save it first."
        )


# ============================================================
# 8. TRAINING
# ============================================================

if start_button:
    try:
        with st.spinner("Preparing the dataset..."):
            loader = get_dataloader(
                source=dataset_source,
                batch_size=batch_size,
                limit=DATA_LIMIT,
            )

        initialize_session(learning_rate)

        progress_bar = st.progress(0)
        status = st.empty()
        loss_area = st.empty()

        for epoch_index in range(epochs):
            status.info(
                f"Training epoch {epoch_index + 1} of {epochs}..."
            )

            results = train_one_epoch(
                st.session_state.generator,
                st.session_state.discriminator,
                loader,
                st.session_state.optimizer_g,
                st.session_state.optimizer_d,
                device=str(DEVICE),
                latent_dim=LATENT_DIM,
                max_batches=(
                    int(max_batches)
                    if int(max_batches) > 0
                    else None
                ),
            )

            st.session_state.epoch_history.append({
                "Epoch": len(st.session_state.epoch_history) + 1,
                "Generator Loss": results["generator_loss"],
                "Discriminator Loss": results["discriminator_loss"],
            })

            progress_bar.progress(
                (epoch_index + 1) / epochs
            )

            history_df = pd.DataFrame(
                st.session_state.epoch_history
            )

            with loss_area.container():
                st.line_chart(
                    history_df.set_index("Epoch")[
                        ["Generator Loss", "Discriminator Loss"]
                    ]
                )

        st.session_state.training_complete = True

        status.success("Training completed successfully!")

        st.session_state.generator.eval()

        with torch.no_grad():
            noise = torch.randn(
                16, LATENT_DIM, 1, 1, device=DEVICE
            )
            st.session_state.generated_images = (
                st.session_state.generator(noise).cpu()
            )

        st.success(
            "Generated sample images are ready. "
            "Use Save Checkpoint to save your trained model."
        )

    except Exception as exc:
        st.error(
            "Training failed. Check the dataset, model dimensions, "
            "and backend function signatures."
        )
        st.exception(exc)


# ============================================================
# 9. SAVE CHECKPOINT
# ============================================================

if save_button:
    if not st.session_state.models_ready:
        st.warning("Train a model or load a checkpoint first.")
    else:
        try:
            save_checkpoint(
                str(CHECKPOINT_PATH),
                st.session_state.generator,
                st.session_state.discriminator,
                st.session_state.optimizer_g,
                st.session_state.optimizer_d,
                epoch=len(st.session_state.epoch_history),
                history=st.session_state.epoch_history,
                latent_dim=LATENT_DIM,
            )

            st.success(
                f"Checkpoint saved successfully: {CHECKPOINT_PATH}"
            )

        except Exception as exc:
            st.error("Checkpoint saving failed.")
            st.exception(exc)


# ============================================================
# 10. GENERATE IMAGES
# ============================================================

if generate_button:
    if not st.session_state.models_ready:
        st.warning("Train a model or load a checkpoint first.")
    else:
        try:
            st.session_state.generator.eval()

            with torch.no_grad():
                noise = torch.randn(
                    16, LATENT_DIM, 1, 1, device=DEVICE
                )
                st.session_state.generated_images = (
                    st.session_state.generator(noise).cpu()
                )

            st.success("New images generated!")

        except Exception as exc:
            st.error("Image generation failed.")
            st.exception(exc)


# ============================================================
# 11. TRAINING HISTORY
# ============================================================

st.divider()
st.header("📊 Training History")

if st.session_state.epoch_history:
    history_df = pd.DataFrame(st.session_state.epoch_history)

    st.line_chart(
        history_df.set_index("Epoch")[
            ["Generator Loss", "Discriminator Loss"]
        ]
    )

    st.dataframe(history_df, use_container_width=True)

    csv_data = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Training History (CSV)",
        data=csv_data,
        file_name="dcgan_training_history.csv",
        mime="text/csv",
    )
else:
    st.info("Training history will appear here after training.")


# ============================================================
# 12. GENERATED IMAGE GALLERY
# ============================================================

st.divider()
st.header("🖼️ Generated Fashion Images")

images = st.session_state.generated_images

if images is not None:
    images = ((images + 1) / 2).clamp(0, 1)

    grid = make_grid(
        images,
        nrow=4,
        padding=3,
    )

    grid_np = grid.permute(1, 2, 0).numpy()

    if grid_np.shape[-1] == 1:
        grid_np = grid_np.squeeze(-1)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(
        grid_np,
        cmap="gray" if grid_np.ndim == 2 else None,
    )
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)

    buffer = io.BytesIO()

    fig_download, ax_download = plt.subplots(figsize=(7, 7))
    ax_download.imshow(
        grid_np,
        cmap="gray" if grid_np.ndim == 2 else None,
    )
    ax_download.axis("off")
    fig_download.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig_download)

    st.download_button(
        "⬇️ Download Generated Images",
        data=buffer.getvalue(),
        file_name="dcgan_fashion_images.png",
        mime="image/png",
    )
else:
    st.info(
        "Generated images will appear here after successful training "
        "or after loading a trained checkpoint."
    )


# ============================================================
# 13. FOOTER
# ============================================================

st.divider()
st.caption(
    "DCGAN Fashion Studio · PyTorch + Streamlit · "
    "Generator and Discriminator"
)
