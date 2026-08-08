import streamlit as st
from urllib.parse import urlparse, parse_qs

from rag import create_rag_chain


# ---------- Page Configuration ----------

st.set_page_config(
    page_title="YouTube RAG Assistant",
    page_icon="🎥",
    layout="wide"
)


# ---------- Header ----------

st.title("🎥 YouTube RAG Assistant")

st.write(
    "Ask questions about a YouTube video using "
    "Retrieval-Augmented Generation (RAG)."
)


# ---------- YouTube URL Helper ----------

def get_video_id(url):

    try:

        parsed_url = urlparse(url)

        # youtu.be/VIDEO_ID
        if parsed_url.hostname == "youtu.be":

            return parsed_url.path[1:]

        # youtube.com/watch?v=VIDEO_ID
        if parsed_url.hostname in [
            "www.youtube.com",
            "youtube.com"
        ]:

            return parse_qs(
                parsed_url.query
            ).get("v", [None])[0]

        return None

    except Exception:

        return None


# ---------- Sidebar ----------

with st.sidebar:

    st.header("⚙️ Settings")

    st.write("RAG Configuration")

    st.info(
        """
Embedding Model:
nvidia/nv-embed-v1

LLM:
meta/llama-3.1-8b-instruct

Retrieved Chunks:
4
"""
    )


# ---------- YouTube Video ----------

st.subheader("📺 YouTube Video")

video_url = st.text_input(
    "Enter YouTube video URL",
    placeholder="https://www.youtube.com/watch?v=..."
)


# ---------- Process Video ----------

if st.button(
    "🔄 Process Video",
    use_container_width=True
):

    if not video_url:

        st.warning(
            "Please enter a YouTube video URL."
        )

    else:

        video_id = get_video_id(video_url)

        if not video_id:

            st.error(
                "Invalid YouTube URL."
            )

        else:

            with st.spinner(
                "Processing YouTube transcript..."
            ):

                try:

                    chain, chunk_count = (
                        create_rag_chain(
                            video_id
                        )
                    )

                    st.session_state.chain = chain
                    st.session_state.video_id = video_id
                    st.session_state.chunk_count = (
                        chunk_count
                    )

                    st.success(
                        "Video processed successfully!"
                    )

                    st.info(
                        f"Created {chunk_count} "
                        "transcript chunks."
                    )

                except Exception as e:

                    st.error(
                        f"Error: {str(e)}"
                    )


# ---------- Question ----------

st.divider()

st.subheader("💬 Ask a Question")

question = st.text_input(
    "Enter your question",
    placeholder="What is this video about?"
)


# ---------- Example Questions ----------

st.write("💡 Example questions:")

col1, col2, col3 = st.columns(3)

with col1:
    st.write(
        "• What is this video about?"
    )

with col2:
    st.write(
        "• What are the main points?"
    )

with col3:
    st.write(
        "• Can you summarize the video?"
    )


# ---------- Ask Question ----------

if st.button(
    "🤖 Ask",
    use_container_width=True
):

    if "chain" not in st.session_state:

        st.warning(
            "Please process a YouTube video first."
        )

    elif not question:

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching transcript and generating answer..."
        ):

            try:

                answer = st.session_state.chain.invoke(
                    question
                )

                st.subheader("🤖 Answer")

                st.write(answer)

            except Exception as e:

                st.error(
                    f"Error: {str(e)}"
                )


# ---------- Footer ----------

st.divider()

st.caption(
    "Built with Streamlit, LangChain, FAISS and NVIDIA AI."
)