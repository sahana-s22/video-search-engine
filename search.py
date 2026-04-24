import cv2
import os
import json
import numpy as np
import faiss
from PIL import Image, ImageDraw, ImageFont

# Load FAISS index and paths
index = faiss.read_index("embeddings/index.faiss")
paths = np.load("embeddings/paths.npy", allow_pickle=True)

def get_image_embedding(image_path=None, image_pil=None, size=256):
    """
    Generate embedding from image using color histogram and basic statistics.
    Matches the embedding generation in index.py exactly.
    """
    if image_path:
        img = cv2.imread(image_path)
    else:
        # Convert PIL image to numpy array (BGR)
        img_np = np.array(image_pil)
        img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
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


def save_results(query, results, output_path="results/results.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "query": query,
        "results": results
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def format_timestamp_from_frame(frame_path):
    filename = os.path.basename(frame_path)
    frame_number = 0

    if filename.startswith("frame_") and filename.endswith(".jpg"):
        try:
            frame_number = int(filename.replace("frame_", "").replace(".jpg", ""))
        except ValueError:
            frame_number = 0

    seconds = frame_number
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def search(query, top_k=5):
    """
    Search for frames matching the query.
    
    Args:
        query: Text query string (e.g., "person walking")
        top_k: Number of top results to return
    
    Returns:
        List of dicts with keys: frame, score, timestamp
    """
    # Create a blank image and write query text on it
    img_pil = Image.new('RGB', (256, 256), color=(73, 109, 137))
    draw = ImageDraw.Draw(img_pil)
    
    # Try to use a default font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Write text on image
    text = query[:30]  # Limit text length
    draw.text((10, 10), text, fill=(255, 255, 255), font=font)
    
    # Generate embedding from this image
    query_embedding = get_image_embedding(image_pil=img_pil)
    query_embedding = query_embedding.reshape(1, -1)
    
    # Search FAISS index
    distances, indices = index.search(query_embedding, top_k)
    
    # Format results
    results = []
    for i, idx in enumerate(indices[0]):
        frame_path = str(paths[idx])
        results.append({
            "frame": frame_path,
            "score": float(distances[0][i]),
            "timestamp": format_timestamp_from_frame(frame_path)
        })

    save_results(query, results)
    return results


if __name__ == "__main__":
    # Test search
    query = "test query"
    results = search(query, top_k=5)
    
    print(f"Search results for: '{query}'")
    for i, res in enumerate(results):
        print(f"{i+1}. {res['frame']} - Score: {res['score']:.4f}")
