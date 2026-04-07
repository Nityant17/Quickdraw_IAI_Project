import os
import requests
from tqdm import tqdm

BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"

files = [
    "circle.npy", "square.npy", "triangle.npy", "hexagon.npy", "line.npy",
    "zigzag.npy", "squiggle.npy", "sun.npy", "moon.npy", "star.npy",
    "cloud.npy", "lightning.npy", "mountain.npy", "tree.npy", "leaf.npy",
    "flower.npy", "tornado.npy", "apple.npy", "banana.npy", "pizza.npy",
    "donut.npy", "ice cream.npy", "mushroom.npy", "watermelon.npy",
    "door.npy", "ladder.npy", "envelope.npy", "cup.npy", "pencil.npy",
    "key.npy", "scissors.npy", "umbrella.npy", "clock.npy", "eyeglasses.npy",
    "book.npy", "hat.npy", "smiley face.npy", "eye.npy", "mustache.npy",
    "hand.npy", "foot.npy", "fish.npy", "snake.npy", "spider.npy",
    "butterfly.npy", "bird.npy", "sailboat.npy", "airplane.npy",
    "car.npy", "sword.npy"
]

output_dir = "./data"
os.makedirs(output_dir, exist_ok=True)

def download_file(filename):
    url = BASE_URL + filename
    local_path = os.path.join(output_dir, filename)

    if os.path.exists(local_path):
        print(f"[SKIP] {filename} already exists")
        return

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        with open(local_path, "wb") as f, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

        print(f"[DONE] {filename}")

    except Exception as e:
        print(f"[ERROR] {filename}: {e}")

for file in files:
    download_file(file)

print("\nAll downloads attempted.")