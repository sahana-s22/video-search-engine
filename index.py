import cv2
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import faiss

def get_image_embedding(image_path, size=256):
    """
    Generate embedding from image using color histogram and basic statistics.
    This avoids downloading any pre-trained models.
    """
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (size, size))
    
    # Convert to LAB color space for better feature representation
    img_lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    
    # Extract color histograms for each channel
    hist_features = []
    for i in range(3):
        hist = cv2.calcHist([img_lab], [i], None, [32], [0, 256])
        hist_features.extend(hist.flatten())
    
    # Extract basic statistics
    stats_features = [
        np.mean(img_lab),
        np.std(img_lab),
        np.min(img_lab),
        np.max(img_lab)
    ]
    
    # Combine all features
    embedding = np.array(hist_features + stats_features, dtype=np.float32)
    
    # Normalize
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    
    return embedding

def extract_frames(video_path, output_folder, fps=1):
    if not os.path.exists(video_path):
        print("❌ Video not found:", video_path)
        return

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Cannot open video. File may be corrupted.")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)

    if video_fps == 0:
        print("❌ FPS not detected. Invalid video.")
        return

    frame_interval = int(video_fps // fps)

    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            filename = f"{output_folder}/frame_{saved}.jpg"
            cv2.imwrite(filename, frame)
            saved += 1

        count += 1

    cap.release()
    print("✅ Frames extracted:", saved)


def generate_embeddings(frame_folder):
    embeddings = []
    paths = []

    files = sorted(os.listdir(frame_folder))

    for file in tqdm(files):
        if file.endswith('.jpg'):
            path = os.path.join(frame_folder, file)
            
            # Get embedding
            emb = get_image_embedding(path)
            
            embeddings.append(emb)
            paths.append(path)

    return np.array(embeddings), paths


def build_and_save_index(embeddings, paths):
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs("embeddings", exist_ok=True)

    faiss.write_index(index, "embeddings/index.faiss")
    np.save("embeddings/paths.npy", paths)

    print("✅ Index saved!")


if __name__ == "__main__":
    video_path = "data/sample.mp4"
    frame_folder = "frames"

    extract_frames(video_path, frame_folder)

    print("🔄 Generating embeddings...")
    embeddings, paths = generate_embeddings(frame_folder)

    print("🔄 Building FAISS index...")
    build_and_save_index(embeddings, paths)

    print("🚀 Indexing complete!")
