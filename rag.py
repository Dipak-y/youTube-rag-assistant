"""
RAG pipeline over YouTube transcript.

Requires .env file with:
NVIDIA_API_KEY=your_key_here
"""

import os
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("NVIDIA_API_KEY"):
    raise RuntimeError(
        "NVIDIA_API_KEY not set. Add it to a .env file."
    )

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from langchain_nvidia_ai_endpoints import (
    NVIDIAEmbeddings,
    ChatNVIDIA
)

from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)

from langchain_core.output_parsers import StrOutputParser


# ---------- Configuration ----------

INDEX_DIR = "faiss_index"


# ---------- Step 1: Get YouTube Transcript ----------

def fetch_transcript(video_id: str, languages=("hi",)) -> str:

    try:
        api = YouTubeTranscriptApi()

        transcript = api.fetch(
            video_id,
            languages=list(languages)
        )

        return " ".join(
            item.text for item in transcript
        )

    except TranscriptsDisabled:
        raise RuntimeError(
            f"Transcripts disabled for video {video_id}"
        )


# ---------- Step 2: Split Transcript ----------

def split_text(text: str):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.create_documents([text])


# ---------- Step 3: Create / Load FAISS ----------

def build_or_load_vector_store(chunks, embeddings):

    if os.path.exists(INDEX_DIR):

        return FAISS.load_local(
            INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )

    store = FAISS.from_documents(
        chunks,
        embeddings
    )

    store.save_local(INDEX_DIR)

    return store


# ---------- Step 4: Format Retrieved Documents ----------

def format_docs(retrieved_docs) -> str:

    return "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )


# ---------- Step 5: Prompt ----------

PROMPT = PromptTemplate(
    template="""
You are a helpful assistant.

Answer ONLY from the provided transcript context.

If the context is insufficient, just say you don't know.

Context:
{context}

Question:
{question}
""",
    input_variables=[
        "context",
        "question"
    ]
)


# ---------- Step 6: Create RAG Chain ----------

def create_rag_chain(
    video_id: str,
    languages=("hi",)
):

    # Fetch transcript
    text = fetch_transcript(
        video_id,
        languages
    )

    print(
        f"Transcript length: {len(text)} characters"
    )

    # Split transcript
    chunks = split_text(text)

    print(
        f"Chunks created: {len(chunks)}"
    )

    # NVIDIA embeddings
    embeddings = NVIDIAEmbeddings(
        model="nvidia/nv-embed-v1"
    )

    # FAISS vector store
    vector_store = build_or_load_vector_store(
        chunks,
        embeddings
    )

    # Retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4
        }
    )

    # NVIDIA LLM
    llm = ChatNVIDIA(
        model="meta/llama-3.1-8b-instruct",
        temperature=0.2
    )

    # Output parser
    parser = StrOutputParser()

    # RAG chain
    chain = (
        RunnableParallel(
            {
                "context": (
                    retriever
                    | RunnableLambda(format_docs)
                ),
                "question": RunnablePassthrough()
            }
        )
        | PROMPT
        | llm
        | parser
    )

    return chain, len(chunks)


# ---------- Optional Terminal Test ----------

if __name__ == "__main__":

    VIDEO_ID = "J5_-l7WIO_w"

    chain, chunks = create_rag_chain(
        VIDEO_ID
    )

    print(
        f"Chunks: {chunks}"
    )

    question = (
        "Is the topic of nuclear fusion "
        "discussed in this video? "
        "If yes, what was discussed?"
    )

    answer = chain.invoke(
        question
    )

    print("\nAnswer:")
    print(answer)