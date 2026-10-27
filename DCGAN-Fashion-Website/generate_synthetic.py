from pathlib import Path
import random
from PIL import Image, ImageDraw


def generate_synthetic_dataset(output_dir="data/synthetic", count=500, seed=42):
    """Create simple synthetic grayscale clothing-like silhouettes for pipeline testing."""
    random.seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        image = Image.new("L", (28, 28), 0)
        draw = ImageDraw.Draw(image)
        shade = random.randint(150, 255)
        kind = random.choice(["shirt", "trousers", "dress", "bag", "shoe"])

        if kind == "shirt":
            draw.polygon([(8, 5), (3, 9), (7, 15), (10, 12), (10, 23),
                          (19, 23), (19, 12), (22, 15), (26, 9),
                          (21, 5), (17, 8), (12, 8)], fill=shade)
        elif kind == "trousers":
            draw.polygon([(7, 4), (21, 4), (20, 12), (18, 24), (13, 24),
                          (14, 13), (12, 13), (11, 24), (6, 24), (7, 12)],
                         fill=shade)
        elif kind == "dress":
            draw.polygon([(10, 4), (18, 4), (19, 10), (24, 23),
                          (4, 23), (9, 10)], fill=shade)
        elif kind == "bag":
            draw.rectangle((6, 10, 22, 24), fill=shade)
            draw.arc((9, 3, 19, 15), 180, 360, fill=shade, width=2)
        else:
            draw.polygon([(4, 17), (13, 17), (18, 20), (25, 21),
                          (25, 24), (4, 24)], fill=shade)

        image.save(output / f"image_{i:04d}.png")
    return str(output)


if __name__ == "__main__":
    print(f"Synthetic dataset saved to {generate_synthetic_dataset()}")
