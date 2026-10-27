
from pathlib import Path


# Checkpoint directory
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Single checkpoint file for both models
CHECKPOINT_PATH = CHECKPOINT_DIR / "dcgan_checkpoint.pth"

import io
import time

import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid




# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DCGAN Fashion Studio",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. PROJECT PATHS AND DEVICE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
SYNTHETIC_DIR = DATA_DIR / "synthetic"

for folder in (DATA_DIR, CHECKPOINT_DIR, SYNTHETIC_DIR):
    folder.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fixed as instructed by your faculty.
DATA_LIMIT = 100


# ============================================================
# 3. CUSTOM CSS
# ============================================================

CUSTOM_CSS = """
<style>
.stApp,
[data-testid="stAppViewContainer"] {
    background-color: #F7F5FF !important;
    color: #30294A !important;
}

[data-testid="stHeader"] {
    background-color: #F7F5FF !important;
}

.main .block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Sidebar */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background-color: #FFFFFF !important;
}

[data-testid="stSidebar"] {
    border-right: 1px solid #E5DEF5 !important;
}

[data-testid="stSidebar"] * {
    color: #30294A;
}

/* Page heading */
.hero-label {
    display: inline-block;
    background-color: #EDE5FF;
    color: #6D28D9;
    padding: 9px 15px;
    border-radius: 25px;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.5px;
    margin-bottom: 12px;
}

.hero-title {
    color: #30294A !important;
    font-size: 42px;
    font-weight: 800;
    line-height: 1.2;
    margin: 8px 0 12px;
}

.hero-description {
    color: #77738E !important;
    font-size: 16px;
    line-height: 1.7;
    margin-bottom: 22px;
}

.section-title {
    color: #30294A !important;
    font-size: 25px;
    font-weight: 750;
    margin: 15px 0 16px;
}

/* Metric cards */
.metric-card {
    background-color: #FFFFFF;
    border: 1px solid #E5DEF5;
    border-radius: 16px;
    padding: 22px;
    min-height: 115px;
    box-shadow: 0 4px 14px rgba(69, 44, 120, 0.05);
}

.metric-label {
    color: #77738E;
    font-size: 13px;
    margin-bottom: 10px;
}

.metric-value {
    color: #6D28D9;
    font-size: 27px;
    font-weight: 800;
    overflow-wrap: anywhere;
}

.info-card {
    background-color: #FFFFFF;
    border: 1px solid #E5DEF5;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 14px;
    color: #30294A;
    line-height: 1.8;
}

/* Lavender dropdowns */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #F1ECFF !important;
    border: 1px solid #D8C7FF !important;
    border-radius: 10px !important;
    min-height: 42px;
    box-shadow: none !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #30294A !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] svg {
    fill: #6D28D9 !important;
    color: #6D28D9 !important;
}

/* Dropdown popup */
[data-baseweb="popover"] [role="listbox"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E5DEF5 !important;
    border-radius: 10px !important;
}

[data-baseweb="popover"] [role="option"] {
    background-color: #FFFFFF !important;
    color: #30294A !important;
}

[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"] {
    background-color: #EDE5FF !important;
    color: #5B21B6 !important;
}

/* Primary buttons: Start Training */
div.stButton > button[kind="primary"] {
    background-color: #6D28D9 !important;
    color: #FFFFFF !important;
    border: 1px solid #6D28D9 !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    min-height: 43px;
    box-shadow: none !important;
    transition: all 0.2s ease;
}

div.stButton > button[kind="primary"]:hover {
    background-color: #5B21B6 !important;
    border-color: #5B21B6 !important;
    color: #FFFFFF !important;
}

/* Secondary buttons */
div.stButton > button[kind="secondary"] {
    background-color: #EDE5FF !important;
    color: #5B21B6 !important;
    border: 1px solid #D8C7FF !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    min-height: 43px;
    box-shadow: none !important;
    transition: all 0.2s ease;
}

div.stButton > button[kind="secondary"]:hover {
    background-color: #DDD0FF !important;
    border-color: #BCA4FF !important;
    color: #4C1D95 !important;
}

div.stButton > button:focus-visible {
    outline: 3px solid #C4B5FD !important;
    outline-offset: 2px !important;
}

/* Purple slider */
[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
    background-color: #6D28D9 !important;
    border-color: #6D28D9 !important;
}

/* Sidebar input fields */
[data-testid="stSidebar"] input {
    background-color: #F1ECFF !important;
    color: #30294A !important;
    border-color: #D8C7FF !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E5DEF5 !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

[data-testid="stMetricLabel"] {
    color: #77738E !important;
}

[data-testid="stMetricValue"] {
    color: #6D28D9 !important;
}

/* Download buttons */
[data-testid="stDownloadButton"] button {
    background-color: #EDE5FF !important;
    color: #5B21B6 !important;
    border: 1px solid #D8C7FF !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}

[data-testid="stDownloadButton"] button:hover {
    background-color: #DDD0FF !important;
    color: #4C1D95 !important;
}

[data-testid="stAlert"] {
    border-radius: 12px !important;
}

hr {
    border-color: #E5DEF5 !important;
}

h1, h2, h3 {
    color: #30294A;
}

.footer {
    text-align: center;
    color: #77738E;
    font-size: 13px;
    line-height: 1.8;
    padding: 15px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# 4. IMPORT BACKEND FUNCTIONS
# ============================================================

try:
    from dcgan import (
        Generator,
        Discriminator,
        initialize_weights,
        LATENT_DIM,
    )

    from dataset import get_dataloader
    from generate_synthetic import generate_synthetic_dataset

    from train import (
        train_one_epoch,
        save_checkpoint,
        load_checkpoint,
    )

except ImportError as exc:
    st.error(
        "Could not import a required backend function. "
        "Check dcgan.py, dataset.py, train.py and "
        "generate_synthetic.py."
    )
    st.exception(exc)
    st.stop()


# ============================================================
# 5. CREATE AND RESET MODELS
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


def reset_training(learning_rate):
    (
        st.session_state.generator,
        st.session_state.discriminator,
        st.session_state.optimizer_g,
        st.session_state.optimizer_d,
    ) = create_models(learning_rate)

    st.session_state.models_ready = True
    st.session_state.epoch_history = []
    st.session_state.training_complete = False
    st.session_state.generated_images = None

    st.session_state.fixed_noise = torch.randn(
        16, LATENT_DIM, 1, 1, device=DEVICE
    )


# ============================================================
# 6. SESSION STATE
# ============================================================

defaults = {
    "models_ready": False,
    "epoch_history": [],
    "training_complete": False,
    "selected_dataset": "Synthetic",
    "generated_images": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# 7. SIDEBAR CONTROLS
# ============================================================

with st.sidebar:
    st.markdown("## ⚙️ Training Controls")
    st.caption("Configure your generative model below.")

    dataset_source = st.selectbox(
        "Choose dataset",
        ["Synthetic", "Fashion-MNIST"],
    )

    epochs = st.slider(
        "Epochs to train",
        min_value=1,
        max_value=20,
        value=2,
        step=1,
    )

    batch_size = st.selectbox(
        "Batch size",
        [16, 32, 64],
        index=0,
    )

    learning_rate = st.selectbox(
        "Learning rate",
        [0.0001, 0.0002, 0.0005, 0.001],
        index=1,
        format_func=lambda value: f"{value:.4f}",
    )

    # Fixed training image limit: 100
    st.markdown("**Training image limit**")
    st.markdown(
        """
        <div style="
            background: #F1ECFF;
            border: 1px solid #D8C7FF;
            border-radius: 10px;
            padding: 12px 14px;
            color: #5B21B6;
            font-weight: 700;
            margin-bottom: 12px;
        ">
            100 images
        </div>
        """,
        unsafe_allow_html=True,
    )

    max_batches = st.selectbox(
        "Batches per epoch",
        [5, 10, 20, 50, 0],
        index=1,
        format_func=lambda value: (
            "All batches" if value == 0 else str(value)
        ),
        help="Choose All batches to use the full data loader.",
    )

    st.divider()
    st.caption(f"**Compute device:** {str(DEVICE).upper()}")

    if DEVICE.type == "cpu":
        st.info("CPU mode detected. Use fewer batches for faster testing.")
    else:
        st.success("GPU detected. CUDA is available.")

    reset_button = st.button(
        "🔄 Reset Model",
        use_container_width=True,
    )

    load_button = st.button(
        "📂 Load Checkpoint",
        use_container_width=True,
    )


# ============================================================
# 8. HANDLE DATASET CHANGES AND RESET
# ============================================================

if st.session_state.selected_dataset != dataset_source:
    st.session_state.selected_dataset = dataset_source
    st.session_state.models_ready = False
    st.session_state.epoch_history = []
    st.session_state.training_complete = False
    st.session_state.generated_images = None

if reset_button:
    reset_training(learning_rate)
    st.success("Model reset successfully.")

if not st.session_state.models_ready:
    reset_training(learning_rate)


# ============================================================
# 9. CHECKPOINT PATHS
# ============================================================

GENERATOR_CHECKPOINT = CHECKPOINT_DIR / "generator.pth"
DISCRIMINATOR_CHECKPOINT = CHECKPOINT_DIR / "discriminator.pth"


# ============================================================
# 10. HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero-label">
        GENERATIVE AI LAB · FASHION DESIGN
    </div>
    <div class="hero-title">
        👕 DCGAN Fashion Studio
    </div>
    <div class="hero-description">
        Train a generative model and explore AI-created
        clothing-like images. Monitor training losses and
        generate new samples from the Generator.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

st.markdown(
    '<div class="section-title">📊 Model Overview</div>',
    unsafe_allow_html=True,
)


# ============================================================
# 11. OVERVIEW CARDS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Selected Dataset</div>
            <div class="metric-value">{dataset_source}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Training Image Limit</div>
            <div class="metric-value">{DATA_LIMIT}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Compute Device</div>
            <div class="metric-value">{str(DEVICE).upper()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 12. PREPARE DATA LOADER
# ============================================================

def prepare_dataloader():
    if dataset_source == "Synthetic":
        existing_images = list(SYNTHETIC_DIR.glob("*.png"))

        if len(existing_images) < DATA_LIMIT:
            with st.spinner("Preparing synthetic training images..."):
                generate_synthetic_dataset(
                    output_dir=str(SYNTHETIC_DIR),
                    count=DATA_LIMIT,
                )

    loader = get_dataloader(
        source=dataset_source,
        batch_size=batch_size,
        limit=DATA_LIMIT,
    )

    return loader


# ============================================================
# 13. LOAD CHECKPOINT
# ============================================================

if load_button:
    if (
        GENERATOR_CHECKPOINT.exists()
        and DISCRIMINATOR_CHECKPOINT.exists()
    ):
        try:
            load_checkpoint(
                str(GENERATOR_CHECKPOINT),
                str(DISCRIMINATOR_CHECKPOINT),
                st.session_state.generator,
                st.session_state.discriminator,
                st.session_state.optimizer_g,
                st.session_state.optimizer_d,
                DEVICE,
            )
            st.success("Checkpoint loaded successfully.")

        except Exception as exc:
            st.error(
                "Checkpoint loading failed. Check the function "
                "signature in train.py."
            )
            st.exception(exc)
    else:
        st.warning("No saved checkpoint found. Train and save the model first.")


# ============================================================
# 14. TRAINING SECTION
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">🚀 Train Your Model</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-card">
        <b>How DCGAN works</b><br>
        The Generator creates images from random noise.
        The Discriminator learns to distinguish real images
        from generated images. Both networks improve through
        adversarial learning.
        <br><br>
        <b>Dataset limit:</b> Training is configured for 100 images.
    </div>
    """,
    unsafe_allow_html=True,
)

train_col, save_col = st.columns([2, 1])

with train_col:
    start_training = st.button(
        "🚀 Start Training",
        type="primary",
        use_container_width=True,
    )

with save_col:
    save_checkpoint_button = st.button(
        "💾 Save Checkpoint",
        use_container_width=True,
    )


# ============================================================
# 15. TRAINING LOOP
# ============================================================

if start_training:
    try:
        loader = prepare_dataloader()

        if loader is None:
            st.error("The data loader could not be created.")
            st.stop()

        progress_bar = st.progress(0)
        status_text = st.empty()
        live_metrics = st.empty()

        for epoch in range(epochs):
            status_text.info(
                f"Training epoch {epoch + 1} of {epochs}..."
            )

            start_time = time.time()

            result = train_one_epoch(
                st.session_state.generator,
                st.session_state.discriminator,
                loader,
                st.session_state.optimizer_g,
                st.session_state.optimizer_d,
                device=DEVICE,
                latent_dim=LATENT_DIM,
                max_batches=(
                    max_batches if max_batches > 0 else None
                ),
            )

            generator_loss = float(result["generator_loss"])
            discriminator_loss = float(result["discriminator_loss"])
            elapsed = time.time() - start_time

            st.session_state.epoch_history.append(
                {
                    "Epoch": len(st.session_state.epoch_history) + 1,
                    "Generator Loss": generator_loss,
                    "Discriminator Loss": discriminator_loss,
                    "Seconds": round(elapsed, 2),
                }
            )

            progress_bar.progress((epoch + 1) / epochs)

            with live_metrics.container():
                metric1, metric2 = st.columns(2)

                metric1.metric(
                    "Generator Loss",
                    f"{generator_loss:.4f}",
                )

                metric2.metric(
                    "Discriminator Loss",
                    f"{discriminator_loss:.4f}",
                )

        st.session_state.training_complete = True

        st.session_state.generator.eval()

        with torch.no_grad():
            st.session_state.generated_images = (
                st.session_state.generator(
                    st.session_state.fixed_noise
                )
                .detach()
                .cpu()
            )

        status_text.success("Training completed successfully!")
        st.success("View your generated images and loss charts below.")

    except Exception as exc:
        st.error(
            "Training failed. Check your backend function signatures, "
            "dataset format, and model output dimensions."
        )
        st.exception(exc)


# ============================================================
# 16. SAVE CHECKPOINT
# ============================================================


if save_checkpoint_button:
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
# 17. TRAINING PERFORMANCE
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">📈 Training Performance</div>',
    unsafe_allow_html=True,
)

if st.session_state.epoch_history:
    history_df = pd.DataFrame(st.session_state.epoch_history)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Generator Loss**")
        st.line_chart(
            history_df.set_index("Epoch")[["Generator Loss"]]
        )

    with chart_col2:
        st.markdown("**Discriminator Loss**")
        st.line_chart(
            history_df.set_index("Epoch")[["Discriminator Loss"]]
        )

    st.markdown("**Training History**")

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Training History (CSV)",
        data=csv_data,
        file_name="dcgan_training_history.csv",
        mime="text/csv",
    )

else:
    st.info(
        "Training history will appear here after you click Start Training."
    )


# ============================================================
# 18. GENERATED FASHION IMAGES
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">🖼️ Generated Fashion Images</div>',
    unsafe_allow_html=True,
)

generated_images = st.session_state.generated_images

if generated_images is not None:
    st.caption(
        "These samples are generated by your trained model. "
        "Their quality depends on training and model architecture."
    )

    images = ((generated_images + 1) / 2).clamp(0, 1)

    grid = make_grid(
        images,
        nrow=4,
        padding=3,
        normalize=False,
    )

    grid_np = grid.permute(1, 2, 0).numpy()

    if grid_np.shape[-1] == 1:
        grid_np = grid_np.squeeze(-1)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(
        grid_np,
        cmap="gray" if grid_np.ndim == 2 else None,
    )
    ax.axis("off")
    fig.tight_layout(pad=0)

    st.pyplot(fig)
    plt.close(fig)

    image_buffer = io.BytesIO()

    fig_download, ax_download = plt.subplots(figsize=(8, 8))
    ax_download.imshow(
        grid_np,
        cmap="gray" if grid_np.ndim == 2 else None,
    )
    ax_download.axis("off")
    fig_download.tight_layout(pad=0)

    fig_download.savefig(
        image_buffer,
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig_download)

    st.download_button(
        "⬇️ Download Generated Image Grid",
        data=image_buffer.getvalue(),
        file_name="dcgan_generated_fashion.png",
        mime="image/png",
    )

else:
    st.info(
        "Generated images will appear here after the first successful "
        "training run."
    )


# ============================================================
# 19. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        <b>DCGAN Fashion Studio</b><br>
        Deep Learning Project · Generator + Discriminator<br>
        Built with PyTorch and Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
