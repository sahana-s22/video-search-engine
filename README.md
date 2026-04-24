# 🎥 Intelligent Video Search Engine

## 🚀 Overview

This project is an AI-powered video search engine that allows users to search video content using natural language queries and retrieve the most relevant frames with timestamps.

---

## 🧠 Architecture

Pipeline:
Video → Frame Extraction → Feature Embedding → FAISS Index → Query Search → UI Display

* Frame Sampling: 1 frame per second
* Feature Extraction: ResNet50 (used as embedding generator)
* Vector Search: FAISS for fast similarity search
* UI: Streamlit

---

## ⚙️ Setup & Installation

```bash
pip install -r requirements.txt
python index.py
streamlit run app.py
```

---

## 🔍 How It Works

1. Video is split into frames
2. Each frame is converted into embeddings
3. Embeddings are stored in FAISS index
4. User query is converted into embedding
5. Similar frames are retrieved and displayed

---

## 📊 Output

Each query returns:

* Frame preview
* Timestamp (HH:MM:SS)
* Relevance score

---

## ⚡ Performance

* Indexing speed: ~300 frames/sec
* Query latency: < 1 second

---

## 🎯 Design Decisions

* Used ResNet50 due to hardware constraints instead of CLIP
* FAISS chosen for fast nearest neighbor search
* Streamlit used for quick UI development

---

## ⚠️ Limitations

* Text understanding is approximate (image-based model)
* Temporal relationships not deeply modeled

---

## 🔥 Future Improvements

* Replace ResNet with CLIP for better semantic understanding
* Add temporal reasoning
* Add video playback with timestamp jump

---

## 🎥 Demo Video

(Add your demo video link here)
# video-search-engine
