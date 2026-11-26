"""
Self-test helper: writes a tiny FAKE dataset with the same folder layout as the two public
datasets, so the whole can be run end-to-end in a few minutes before the real data is
downloaded. 

  python make_synthetic_data.py --out /tmp/fake
"""
import argparse, os
import numpy as np
from PIL import Image, ImageDraw

SPECIES = {"Rohu": (150, 150, 165), "Katla": (120, 130, 110), "Mrigal": (170, 160, 140),
           "Grass Carp": (110, 120, 90), "Silver Carp": (200, 200, 205), "Common Carp": (160, 120, 60)}
HEALTH = ["Healthy Fish", "Bacterial Red Disease", "EUS", "Bacterial Gill Disease"]


def fish(rng, body, disease, w=400, h=300):
    im = Image.new("RGB", (w, h), tuple(int(v) for v in rng.integers(60, 200, 3)))
    d = ImageDraw.Draw(im)
    cx, cy = w // 2 + rng.integers(-30, 30), h // 2 + rng.integers(-20, 20)
    L, H = rng.integers(130, 170), rng.integers(45, 65)
    col = tuple(int(np.clip(c + rng.integers(-15, 15), 0, 255)) for c in body)
    d.ellipse([cx - L, cy - H, cx + L, cy + H], fill=col)
    d.polygon([(cx + L - 10, cy), (cx + L + 50, cy - 40), (cx + L + 50, cy + 40)], fill=col)
    d.ellipse([cx - L + 25, cy - 12, cx - L + 40, cy + 3], fill=(20, 20, 20))
    if disease == "Bacterial Red Disease":
        for _ in range(6):
            x, y = cx + rng.integers(-L + 40, L - 20), cy + rng.integers(-H + 10, H - 10)
            d.ellipse([x - 9, y - 6, x + 9, y + 6], fill=(200, 30, 30))
    elif disease == "EUS":
        for _ in range(2):
            x, y = cx + rng.integers(-L + 50, L - 30), cy + rng.integers(-H + 15, H - 15)
            d.ellipse([x - 16, y - 12, x + 16, y + 12], fill=(230, 220, 210), outline=(120, 20, 20), width=4)
    elif disease == "Bacterial Gill Disease":
        d.ellipse([cx - L + 45, cy - 25, cx - L + 75, cy + 25], fill=(140, 90, 60))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    for sp, body in SPECIES.items():                       # species-only "market" set
        d = os.path.join(a.out, "BD-Freshwater-Fish", sp); os.makedirs(d, exist_ok=True)
        for k in range(a.n):
            fish(rng, body, None).save(os.path.join(d, f"{sp}_{k}.jpg"))
    for he in HEALTH:                                       # species x health "farm" set, specimen folders
        for sp in ("Rohu", "Grass Carp", "Silver Carp"):
            for s in range(a.n // 6):
                d = os.path.join(a.out, "MatsyaDx-BD", he, sp, f"specimen_{s:03d}"); os.makedirs(d, exist_ok=True)
                for v in range(3):                           # 3 views of the same fish
                    fish(rng, SPECIES[sp], he).save(os.path.join(d, f"view{v}.jpg"))
    print("fake data written to", a.out)


if __name__ == "__main__":
    main()
