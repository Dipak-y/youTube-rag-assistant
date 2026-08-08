# 🎥 YouTube RAG Assistant

A Streamlit-based Retrieval-Augmented Generation (RAG) application that allows users to ask questions about the content of a YouTube video.

The application retrieves the video's transcript, divides it into smaller chunks, converts the chunks into vector embeddings, stores them using FAISS, and retrieves the most relevant information when a user asks a question. NVIDIA AI models are then used to generate an answer based only on the retrieved transcript context.

---

## 🚀 Features

- 🎥 Accepts YouTube video URLs
- 📝 Extracts YouTube transcripts
- ✂️ Splits transcripts into smaller chunks
- 🧠 Generates embeddings using NVIDIA AI
- 🔎 Performs similarity-based retrieval using FAISS
- 🤖 Generates answers using NVIDIA Llama
- 💬 Interactive question-answer interface
- 🖥️ Simple and responsive Streamlit UI
- ⚡ Stores the FAISS index locally to reduce repeated processing
- 🔐 Keeps the NVIDIA API key in a `.env` file

---

## 🛠️ Technologies Used

- **Python**
- **Streamlit**
- **LangChain**
- **FAISS**
- **YouTube Transcript API**
- **NVIDIA AI**
- **NVIDIA Embeddings**
- **Meta Llama 3.1**
- **python-dotenv**

---

## 🏗️ Project Architecture

```text
             YouTube Video URL
                     │
                     ▼
          YouTube Transcript API
                     │
                     ▼
             Transcript Text
                     │
                     ▼
              Text Splitting
                     │
                     ▼
           NVIDIA Embeddings
                     │
                     ▼
                FAISS
            Vector Database
                     │
                     ▼
              User Question
                     │
                     ▼
               Similarity
                Retrieval
                     │
                     ▼
           Relevant Transcript
                Context
                     │
                     ▼
             NVIDIA Llama
                     │
                     ▼
               Final Answer
                     │
                     ▼
              Streamlit UI
