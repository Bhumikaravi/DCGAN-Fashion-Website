from pathlib import Path
import random
from PIL import Image, ImageDraw, ImageFilter


def draw_shirt(draw, shade):
    """Draw a detailed T-shirt / shirt."""
    # Main body
    draw.polygon([(8, 6), (20, 6), (20, 24), (8, 24)], fill=shade)
    # Left sleeve
    draw.polygon([(8, 6), (2, 11), (5, 15), (8, 12)], fill=shade)
    # Right sleeve
    draw.polygon([(20, 6), (26, 11), (23, 15), (20, 12)], fill=shade)
    # Crew neck / collar
    draw.arc((11, 4, 17, 9), 0, 180, fill=0, width=1)
    # Hem and sleeve creases
    draw.line([(8, 24), (20, 24)], fill=max(0, shade - 30), width=1)


def draw_trousers(draw, shade):
    """Draw a detailed pair of pants / trousers."""
    # Waistband
    draw.rectangle([7, 4, 21, 6], fill=shade)
    # Left leg
    draw.polygon([(7, 6), (13, 6), (12, 25), (6, 25)], fill=shade)
    # Right leg
    draw.polygon([(15, 6), (21, 6), (22, 25), (16, 25)], fill=shade)
    # Crotch gap
    draw.polygon([(13, 12), (14, 11), (15, 12), (15, 25), (13, 25)], fill=0)


def draw_dress(draw, shade):
    """Draw a stylish dress / gown."""
    # Bodice
    draw.polygon([(10, 4), (18, 4), (19, 11), (9, 11)], fill=shade)
    # Flared A-line skirt
    draw.polygon([(9, 11), (19, 11), (25, 25), (3, 25)], fill=shade)
    # Neckline
    draw.arc((12, 3, 16, 7), 0, 180, fill=0, width=1)
    # Waistline belt accent
    draw.line([(9, 11), (19, 11)], fill=min(255, shade + 35), width=1)


def draw_coat(draw, shade):
    """Draw an overcoat / jacket with lapels."""
    # Coat body
    draw.polygon([(7, 5), (21, 5), (22, 25), (6, 25)], fill=shade)
    # Long sleeves
    draw.polygon([(7, 5), (2, 12), (4, 23), (7, 21)], fill=shade)
    draw.polygon([(21, 5), (26, 12), (24, 23), (21, 21)], fill=shade)
    # Open collar / lapels
    draw.polygon([(11, 5), (14, 13), (17, 5)], fill=0)
    # Center zipper / button line
    draw.line([(14, 13), (14, 25)], fill=0, width=1)


def draw_sneaker(draw, shade):
    """Draw a sneaker / running shoe."""
    # Upper
    draw.polygon([(4, 17), (10, 14), (16, 14), (20, 18), (26, 19), (26, 22), (4, 22)], fill=shade)
    # Collar / ankle opening
    draw.ellipse([(8, 12), (15, 16)], fill=shade)
    # Thick Sole
    draw.rectangle([3, 22, 26, 25], fill=min(255, shade + 40))


def draw_bag(draw, shade):
    """Draw a fashion handbag / tote bag."""
    # Bag body
    draw.rounded_rectangle([6, 10, 22, 24], radius=3, fill=shade)
    # Rounded Handle
    draw.arc((10, 4, 18, 14), 180, 360, fill=shade, width=2)
    # Buckle / clasp detail
    draw.rectangle([13, 14, 15, 16], fill=min(255, shade + 50))


def generate_synthetic_dataset(output_dir="data/synthetic", count=500, seed=42):
    """Create realistic grayscale fashion silhouettes and textures."""
    random.seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    draw_funcs = [
        draw_shirt,
        draw_trousers,
        draw_dress,
        draw_coat,
        draw_sneaker,
        draw_bag,
    ]

    for i in range(count):
        image = Image.new("L", (28, 28), 0)
        draw = ImageDraw.Draw(image)
        shade = random.randint(160, 250)

        chosen_func = random.choice(draw_funcs)
        chosen_func(draw, shade)

        # Subtle anti-aliasing / smoothing for natural fabric look
        if random.random() > 0.5:
            image = image.filter(ImageFilter.SMOOTH_MORE)

        image.save(output / f"image_{i:04d}.png")

    return str(output)


if __name__ == "__main__":
    print(f"Synthetic dataset saved to {generate_synthetic_dataset()}")
