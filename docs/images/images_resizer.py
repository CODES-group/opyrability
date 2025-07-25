from PIL import Image
from pathlib import Path

# Root directory (where this script and images are)
ROOT = Path().resolve()
THUMBS = ROOT / "thumbs"
THUMBS.mkdir(exist_ok=True)

# Target image width
TARGET_W = 300

# Explicitly name all the image files you're trying to resize
# images = [
#     "shower_oi.gif",
#     "dma-mr.PNG",
#     "m-DAC_ProcessPFD.png",
#     "aspen_flowsheet.png",
#     "unisim_flowsheet.png",
#     "H_SOEC.png",
# ]
images = [
    "air_cooling.png",
]

# Resize each image
for name in images:
    img_path = ROOT / name
    if not img_path.exists():
        print(f"❌ File not found: {img_path}")
        continue

    try:
        im = Image.open(img_path)
        w, h = im.size
        new_h = int(h * TARGET_W / w)
        im = im.resize((TARGET_W, new_h), Image.LANCZOS)
        im.save(THUMBS / name)
        print(f"✅ Resized: {name}")
    except Exception as e:
        print(f"⚠️ Error processing {name}: {e}")