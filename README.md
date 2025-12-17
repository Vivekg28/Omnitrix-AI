# 👥 Team Members

* **[Vivek Gautam]** - SID: [202418053]
* **[Bhargav Pachouli]** - SID: [202418009]
* **[Chandan Pandit]** - SID: [202418043]
* **[Nikhil Kumar Sharma]** - SID: [202418039]

---

# ⌚ Ben 10 Omnitrix RAG

A local, multimodal AI system that acts as a real-life Omnitrix. This project uses **Retrieval-Augmented Generation (RAG)** to identify Ben 10 aliens from images and find aliens based on text descriptions.

![Project Banner](https://img.shields.io/badge/Omnitrix-Active-brightgreen?style=for-the-badge) ![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge) ![AI](https://img.shields.io/badge/AI-Multimodal-orange?style=for-the-badge)

## 📋 Overview

The **Omnitrix RAG** is a dual-modality search engine designed to analyze and retrieve information about Ben 10 characters. It bridges the gap between **Visual Data** (images) and **Semantic Data** (descriptions/lore) without relying on external cloud APIs.

**Core Capabilities:**
* **🔍 Search Mode (Text-to-Image):** Describe an alien (e.g., *"Red fire guy made of rocks"*), and the AI retrieves the correct character (Heatblast) by understanding the meaning of your text.
* **🧬 Scan Mode (Image-to-Text):** Upload an image of an alien. The system identifies them using visual matching and generates a tactical description using a Vision LLM.

## ✨ Features

* **Local Privacy:** Runs 100% offline using Ollama and local vector databases.
* **Hybrid RAG:** Combines text embeddings (Description + Powers) and image embeddings (Pixel Data) in a single **ChromaDB** knowledge base.
* **Vision Intelligence:** Uses **Moondream** (via Ollama) to analyze user uploads and provide detailed observations.
* **Omnitrix UI:** A custom-themed **Streamlit** interface with a green/black hacker aesthetic.

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Vector Database:** ChromaDB
* **Embedding Model:** OpenCLIP (ViT-B-32)
* **Vision Model:** Moondream (via Ollama)
* **Backend Logic:** Python, Pandas, NumPy

## 🚀 Installation

### Prerequisites
1.  **Python 3.8+** installed.
2.  **Ollama** installed and running.
    * Pull the vision model: `ollama pull moondream`
