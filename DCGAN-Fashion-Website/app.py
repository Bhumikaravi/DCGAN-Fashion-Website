
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
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 2. STYLING - PURPLE THEME WITH READABLE TEXT
# ============================================================

st.markdown(
    """
    <style>
    /* Main application background */
    .stApp {
        background: linear-gradient(
            135deg,
            #faf7ff,
            #f2edff
        );
        color: #29213d;
    }

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(
            135deg,
            #faf7ff,
            #f2edff
        );
    }

    /* Main headings */
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3 {
        color: #39205f !important;
    }

    /* Main text, labels and captions */
    [data-testid="stMain"] p,
    [data-testid="stMain"] label,
    [data-testid="stMain"] [data-testid="stCaptionContainer"] {
        color: #49365f !important;
    }

    /* Purple gradient hero banner */
    .hero {
        background: linear-gradient(
            120deg,
            #6d28d9,
            #a78bfa
        );
        color: #ffffff;
        padding: 28px;
        border-radius: 18px;
        margin-bottom: 20px;
        box-shadow: 0 5px 18px rgba(109, 40, 217, 0.15);
    }

    .hero h1 {
        color: #ffffff !important;
        margin-bottom: 8px;
    }

    .hero p {
        color: #ffffff !important;
    }

    /* KPI metric cards */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid #ded0fa;
        border-radius: 12px;
        padding: 16px;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] *,
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {
        color: #39205f !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #262630;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: #ffffff !important;
    }

    /* Sidebar help and information text */
    [data-testid="stSidebar"] [data-testid="stAlert"] p {
        color: inherit !important;
    }

    /* Buttons */
    [data-testid="stButton"] button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
        padding: 0.5rem 1rem;
    }

    [data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(
            120deg,
            #6d28d9,
            #8b5cf6
        ) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(109, 40, 217, 0.25);
    }

    [data-testid="stButton"] button[kind="secondary"] {
        background: #ffffff;
        color: #39205f;
        border: 1px solid #c4b5fd;
    }

    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"] {
        background: #343440;
        color: #ffffff !important;
        border: 1px solid #585070;
    }

    [data-testid="stButton"] button:hover {
        border-color: #8b5cf6 !important;
        transform: translateY(-1px);
    }

    /* Alerts */
    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* Download buttons */
    [data-testid="stDownloadButton"] button {
        border-radius: 10px;
        border: 1px solid #8b5cf6;
    }

    /* Tables */
    [data-testid="stDataFrame"] {
        border: 1px solid #ded0fa;
        border-radius: 10px;
    }

    /* Small note */
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

    return (
        generator,
        discriminator,
        optimizer_g,
        optimizer_d,
    )


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


# Initialize session variables
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
        <p>
            Train a Deep Convolutional Generative Adversarial
            Network and generate new fashion-like images using AI.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    f"Compute device: {str(DEVICE).upper()} | "
    f"Training image limit: {DATA_LIMIT}"
)


# ============================================================
# 5. SIDEBAR TRAINING CONTROLS
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

    st.caption(
        "ℹ️ Fashion-MNIST downloads automatically. "
        "Synthetic mode requires image files."
    )

    side_start = st.button(
        "🚀 Start Training",
        use_container_width=True,
        type="primary",
        key="side_start",
    )

    side_generate = st.button(
        "🖼️ Generate Images",
        use_container_width=True,
        key="side_generate",
    )

    side_save = st.button(
        "💾 Save Checkpoint",
        use_container_width=True,
        key="side_save",
    )

    side_load = st.button(
        "📂 Load Checkpoint",
        use_container_width=True,
        key="side_load",
    )


# ============================================================
# 6. MODEL OVERVIEW & MAIN ACTION CONTROLS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Dataset",
        value=dataset_source,
    )

with col2:
    st.metric(
        label="Image Limit",
        value=DATA_LIMIT,
    )

with col3:
    st.metric(
        label="Compute Device",
        value=str(DEVICE).upper(),
    )

st.markdown("### ⚡ Quick Actions")
act_col1, act_col2, act_col3, act_col4 = st.columns(4)

with act_col1:
    main_start = st.button(
        "🚀 Start Training",
        type="primary",
        use_container_width=True,
        key="main_start",
    )

with act_col2:
    main_generate = st.button(
        "🖼️ Generate Images",
        use_container_width=True,
        key="main_generate",
    )

with act_col3:
    main_save = st.button(
        "💾 Save Checkpoint",
        use_container_width=True,
        key="main_save",
    )

with act_col4:
    main_load = st.button(
        "📂 Load Checkpoint",
        use_container_width=True,
        key="main_load",
    )

start_button = main_start or side_start
generate_button = main_generate or side_generate
save_button = main_save or side_save
load_button = main_load or side_load


# ============================================================
# 7. LOAD CHECKPOINT
# ============================================================

if load_button:
    if CHECKPOINT_PATH.exists():
        try:
            if not st.session_state.models_ready:
                initialize_session(learning_rate)

            (
                saved_epoch,
                saved_history,
            ) = load_checkpoint(
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
                "Checkpoint loaded successfully. "
                f"Saved epoch: {saved_epoch}"
            )

        except Exception as exc:
            st.error("Could not load the checkpoint.")
            st.exception(exc)

    else:
        st.warning(
            "No checkpoint found. Train the model and save it first."
        )


# ============================================================
# 8. TRAINING
# ============================================================

if start_button:
    try:
        # Prepare the selected dataset
        with st.spinner("Preparing the dataset..."):
            loader = get_dataloader(
                source=dataset_source,
                batch_size=batch_size,
                limit=DATA_LIMIT,
            )

        # Initialize models
        initialize_session(learning_rate)

        progress_bar = st.progress(0)
        status = st.empty()
        loss_area = st.empty()

        # Train the DCGAN
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

            st.session_state.epoch_history.append(
                {
                    "Epoch": len(
                        st.session_state.epoch_history
                    ) + 1,
                    "Generator Loss": results[
                        "generator_loss"
                    ],
                    "Discriminator Loss": results[
                        "discriminator_loss"
                    ],
                }
            )

            progress_bar.progress(
                (epoch_index + 1) / epochs
            )

            history_df = pd.DataFrame(
                st.session_state.epoch_history
            )

            with loss_area.container():
                st.subheader("Live Training Loss")

                st.line_chart(
                    history_df.set_index("Epoch")[
                        [
                            "Generator Loss",
                            "Discriminator Loss",
                        ]
                    ]
                )

        # Mark training as complete
        st.session_state.training_complete = True

        status.success(
            "Training completed successfully!"
        )

        # Generate a sample gallery
        st.session_state.generator.eval()

        with torch.no_grad():
            noise = torch.randn(
                16,
                LATENT_DIM,
                1,
                1,
                device=DEVICE,
            )

            st.session_state.generated_images = (
                st.session_state.generator(noise).cpu()
            )

        st.success(
            "Sample images are ready! "
            "Use Save Checkpoint to save the trained model."
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
        st.warning(
            "Train a model or load a checkpoint first."
        )

    else:
        try:
            save_checkpoint(
                str(CHECKPOINT_PATH),
                st.session_state.generator,
                st.session_state.discriminator,
                st.session_state.optimizer_g,
                st.session_state.optimizer_d,
                epoch=len(
                    st.session_state.epoch_history
                ),
                history=st.session_state.epoch_history,
                latent_dim=LATENT_DIM,
            )

            st.success(
                "Checkpoint saved successfully."
            )

        except Exception as exc:
            st.error("Checkpoint saving failed.")
            st.exception(exc)


# ============================================================
# 10. GENERATE IMAGES
# ============================================================

if generate_button:
    if not st.session_state.models_ready:
        st.warning(
            "Train a model or load a checkpoint first."
        )

    else:
        try:
            st.session_state.generator.eval()

            with torch.no_grad():
                noise = torch.randn(
                    16,
                    LATENT_DIM,
                    1,
                    1,
                    device=DEVICE,
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
    history_df = pd.DataFrame(
        st.session_state.epoch_history
    )

    st.line_chart(
        history_df.set_index("Epoch")[
            [
                "Generator Loss",
                "Discriminator Loss",
            ]
        ]
    )

    st.dataframe(
        history_df,
        use_container_width=True,
    )

    csv_data = history_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Training History (CSV)",
        data=csv_data,
        file_name="dcgan_training_history.csv",
        mime="text/csv",
    )

else:
    st.info(
        "Training history will appear here after training."
    )


# ============================================================
# 12. GENERATED FASHION IMAGE GALLERY
# ============================================================

st.divider()
st.header("🖼️ Generated Fashion Images")

images = st.session_state.generated_images

if images is not None:
    # Convert image values from [-1, 1] to [0, 1]
    images = ((images + 1) / 2).clamp(0, 1)

    grid = make_grid(
        images,
        nrow=4,
        padding=3,
    )

    grid_np = grid.permute(1, 2, 0).numpy()

    if grid_np.shape[-1] == 1:
        grid_np = grid_np.squeeze(-1)

    # Display image grid
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(
        grid_np,
        cmap="gray" if grid_np.ndim == 2 else None,
    )

    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)

    # Create downloadable PNG
    buffer = io.BytesIO()

    fig_download, ax_download = plt.subplots(
        figsize=(7, 7)
    )

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
        "Generated images will appear here after successful "
        "training or after loading a trained checkpoint."
    )


# ============================================================
# 13. FOOTER
# ============================================================

st.divider()

st.caption(
    "DCGAN Fashion Studio · PyTorch + Streamlit · "
    "Generator and Discriminator"
)
