**ICD-10 Coding Assistant**

**Inference Analytics AI – Capstone Project**

**📘 Overview**

This project develops an AI-powered assistant that predicts ICD-10 medical codes from short clinical text descriptions.
The goal is to make the medical coding process faster, easier, and more consistent for healthcare professionals.

The system combines retrieval-based search and large language models (LLMs) to improve context understanding and code accuracy.


**⚙️ How It Works**

🩺 Data Collection: ICD-10 code descriptions and related medical text were gathered from public sources.

💾 Knowledge Base: All ICD-10 data is embedded using transformer models and stored in Milvus, a vector database that enables fast and efficient similarity search.

🧠 Model: A fine-tuned Qwen-based Large Language Model (LLM) uses the retrieved information to predict the most relevant ICD-10 codes.

🔍 RAG Pipeline: The system applies a Retrieval-Augmented Generation (RAG) approach:

Retrieve the most relevant ICD-10 entries from the vector database

Generate and refine predictions based on the retrieved context


**✨ Key Features**

Predicts ICD-10 codes from short clinical notes or diagnosis descriptions

Uses a knowledge-augmented retrieval system for context-aware predictions

Adaptable for clinical documentation or healthcare analytics tools

Modular design for easy integration and model fine-tuning


**🧰 Tech Stack**

Python 3.10+

Sentence-Transformers for embeddings

Milvus for vector search and retrieval

Hugging Face Transformers for reranking and LLM integration

PyTorch, pandas, tqdm for data handling and processing


**🚀 Setup**

1. Clone the Repository

git clone https://github.com/SwayamDesai/icd10-coding-assistant.git
cd icd10-coding-assistant


2. Install Dependencies

pip install sentence-transformers transformers accelerate pandas tqdm torch pymilvus bitsandbytes


3. Run the Pipeline

python main.py


**🎯 Purpose**

Medical coding is a crucial but time-intensive process.
By combining retrieval-based search and large language models, this project demonstrates how AI can make coding workflows more efficient, accurate, and consistent—ultimately supporting better healthcare data management and decision-making.
