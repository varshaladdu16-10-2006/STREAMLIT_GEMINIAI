import streamlit as st
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
import os
 
 
# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------
 
load_dotenv()
 
API_KEY = os.getenv("GEMINI_API_KEY")
 
if not API_KEY:
    st.error("GEMINI_API_KEY is missing in the .env file.")
    st.stop()
 
 
# --------------------------------------------------
# GEMINI CLIENT
# --------------------------------------------------
 
client = genai.Client(api_key=API_KEY)
 
 
# --------------------------------------------------
# CHROMADB
# --------------------------------------------------
 
chroma_client = chromadb.Client()
 
collection = chroma_client.get_or_create_collection(
    name="gemini_rag_collection"
)
 
 
# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
 
st.set_page_config(
    page_title="Gemini RAG",
    page_icon="📚",
    layout="wide"
)
 
 
# --------------------------------------------------
# CUSTOM CSS  (styling only - nothing else changed)
# --------------------------------------------------
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,800&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Hanken+Grotesk:wght@400;500;700&display=swap');
 
:root {
    --paper: #EEF1F7;
    --card: #FFFFFF;
    --ink: #16213E;
    --ink-soft: #55607A;
    --marker: #FFD84D;
    --signal: #2F5BEA;
    --line: #D3D9E6;
}
 
/* ---------- Base ---------- */
.stApp {
    background-color: var(--paper);
    background-image: radial-gradient(var(--line) 1px, transparent 1px);
    background-size: 22px 22px;
    color: var(--ink);
    font-family: 'Hanken Grotesk', sans-serif;
}
 
.block-container {
    max-width: 860px;
    padding-top: 3.5rem;
    padding-bottom: 5rem;
}
 
#MainMenu, footer { visibility: hidden; }
 
[data-testid="stMarkdownContainer"] p,
label, li { color: var(--ink); }
 
/* ---------- Title ---------- */
h1 {
    font-family: 'Bricolage Grotesque', sans-serif !important;
    font-weight: 800 !important;
    font-size: clamp(2.2rem, 5.5vw, 3.4rem) !important;
    line-height: 1.05 !important;
    letter-spacing: -0.03em !important;
    color: var(--ink) !important;
    padding-bottom: 0.5rem !important;
}
 
h1::after {
    content: "";
    display: block;
    width: 5.5rem;
    height: 0.55rem;
    margin-top: 0.7rem;
    background: var(--marker);
    border-radius: 2px;
    transform: rotate(-1.2deg);
}
 
/* ---------- Other headings ---------- */
h2, h3 {
    font-family: 'Bricolage Grotesque', sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
    color: var(--ink) !important;
}
 
/* ---------- File uploader ---------- */
[data-testid="stFileUploaderDropzone"] {
    background: var(--card);
    border: 2px dashed var(--ink);
    border-radius: 14px;
    padding: 1.6rem;
    transition: border-color 0.15s ease, background 0.15s ease;
}
 
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--signal);
    background: #F7F9FF;
}
 
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span {
    color: var(--ink-soft);
}
 
/* ---------- Buttons ---------- */
div.stButton > button {
    background: var(--ink);
    color: #FFFFFF;
    border: 2px solid var(--ink);
    border-radius: 10px;
    padding: 0.65rem 1.4rem;
    font-family: 'Hanken Grotesk', sans-serif;
    font-weight: 700;
    letter-spacing: 0.01em;
    box-shadow: 4px 4px 0 var(--marker);
    transition: transform 0.12s ease, box-shadow 0.12s ease;
}
 
div.stButton > button p { color: #FFFFFF; }
 
div.stButton > button:hover {
    background: var(--ink);
    border-color: var(--ink);
    color: #FFFFFF;
    transform: translate(-2px, -2px);
    box-shadow: 6px 6px 0 var(--marker);
}
 
div.stButton > button:active {
    transform: translate(3px, 3px);
    box-shadow: 1px 1px 0 var(--marker);
}
 
div.stButton > button:focus-visible {
    outline: 3px solid var(--signal);
    outline-offset: 3px;
}
 
/* ---------- Text input ---------- */
[data-testid="stTextInput"] label p {
    font-weight: 700;
    color: var(--ink);
}
 
[data-testid="stTextInput"] input {
    background: var(--card);
    border: 2px solid var(--ink);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-size: 1.05rem;
    color: var(--ink);
}
 
[data-testid="stTextInput"] input:focus {
    border-color: var(--signal);
    box-shadow: 0 0 0 4px rgba(47, 91, 234, 0.18);
}
 
[data-testid="stTextInput"] > div > div {
    border: none;
    background: transparent;
}
 
/* ---------- Alerts ---------- */
[data-testid="stAlert"] {
    background: var(--card);
    border: 1px solid var(--line);
    border-left: 5px solid var(--signal);
    border-radius: 10px;
    color: var(--ink);
}
 
hr { border-color: var(--line); }
 
/* ---------- Retrieved context (expander) ---------- */
[data-testid="stExpander"] {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 12px;
}
 
[data-testid="stExpander"] summary p {
    font-weight: 700;
    color: var(--ink);
}
 
/* "Chunk 1", "Chunk 2" labels become highlighter tags */
[data-testid="stExpander"] h3 {
    display: inline-block;
    background: var(--marker);
    font-size: 0.85rem !important;
    padding: 0.15rem 0.6rem !important;
    border-radius: 3px;
    margin-top: 0.8rem;
}
 
/* chunk text reads like a quoted passage */
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 0.98rem;
    line-height: 1.65;
    border-left: 5px solid var(--signal);
    background: #FBFCFF;
    padding: 0.8rem 1rem;
    border-radius: 0 8px 8px 0;
    max-width: 72ch;
}
 
/* ---------- Final answer card ----------
   The answer is the last text block on the page, so it is
   targeted by position (works without changing any Python). */
.block-container
[data-testid="stVerticalBlock"] >
[data-testid="stElementContainer"]:last-child:has(> [data-testid="stMarkdown"]),
.block-container
[data-testid="stVerticalBlock"] >
.element-container:last-child:has(> [data-testid="stMarkdown"]) {
    background: var(--card);
    border: 2px solid var(--ink);
    border-radius: 14px;
    padding: 1.4rem 1.7rem;
    box-shadow: 6px 6px 0 var(--marker);
    margin-top: 0.4rem;
}
 
.block-container
[data-testid="stVerticalBlock"] >
[data-testid="stElementContainer"]:last-child:has(> [data-testid="stMarkdown"]) p,
.block-container
[data-testid="stVerticalBlock"] >
.element-container:last-child:has(> [data-testid="stMarkdown"]) p {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.12rem;
    line-height: 1.75;
    max-width: 68ch;
}
 
/* ---------- Accessibility ---------- */
@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; }
}
 
@media (max-width: 640px) {
    .block-container { padding-top: 2rem; }
}
</style>
""", unsafe_allow_html=True)
 
 
# --------------------------------------------------
# TITLE
# --------------------------------------------------
 
st.title("📚 Chat with your PDF")
st.write("RAG using Gemini API + ChromaDB")
 
 
# --------------------------------------------------
# PDF TEXT EXTRACTION
# --------------------------------------------------
 
def extract_text_from_pdf(pdf_file):
 
    reader = PdfReader(pdf_file)
 
    text = ""
 
    for page in reader.pages:
 
        page_text = page.extract_text()
 
        if page_text:
            text += page_text + "\n"
 
    return text
 
 
# --------------------------------------------------
# TEXT CHUNKING
# --------------------------------------------------
 
def create_chunks(text, chunk_size=1000):
 
    chunks = []
 
    start = 0
 
    while start < len(text):
 
        end = start + chunk_size
 
        chunk = text[start:end]
 
        if chunk.strip():
            chunks.append(chunk)
 
        start = end
 
    return chunks
 
 
# --------------------------------------------------
# GEMINI EMBEDDING
# --------------------------------------------------
 
def create_embedding(text):
 
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
 
    return response.embeddings[0].values
 
 
# --------------------------------------------------
# STORE CHUNKS IN CHROMADB
# --------------------------------------------------
 
def store_chunks(chunks):
 
    # Clear previous document
    try:
        chroma_client.delete_collection("gemini_rag_collection")
    except:
        pass
 
    global collection
 
    collection = chroma_client.get_or_create_collection(
        name="gemini_rag_collection"
    )
 
    embeddings = []
 
    for chunk in chunks:
 
        embedding = create_embedding(chunk)
 
        embeddings.append(embedding)
 
    ids = []
 
    for i in range(len(chunks)):
        ids.append(f"chunk_{i}")
 
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )
 
    return len(chunks)
 
 
# --------------------------------------------------
# ASK GEMINI
# --------------------------------------------------
 
def generate_answer(question, context):
 
    prompt = f"""
You are a helpful AI assistant.
 
Answer the user's question using ONLY the information
provided in the context below.
 
If the answer is not available in the context,
say:
 
"I could not find the answer in the uploaded document."
 
Do not make up information.
 
CONTEXT:
{context}
 
QUESTION:
{question}
 
ANSWER:
"""
 
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )
 
    return response.text
 
 
# --------------------------------------------------
# PDF UPLOAD
# --------------------------------------------------
 
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)
 
 
# --------------------------------------------------
# PROCESS PDF
# --------------------------------------------------
 
if uploaded_file is not None:
 
    st.success(f"Uploaded: {uploaded_file.name}")
 
    if st.button("🔄 Process PDF"):
 
        with st.spinner("Reading PDF..."):
 
            text = extract_text_from_pdf(uploaded_file)
 
        if not text.strip():
 
            st.error(
                "Could not extract text from this PDF."
            )
 
        else:
 
            st.info(
                f"Extracted {len(text)} characters."
            )
 
            with st.spinner("Creating chunks..."):
 
                chunks = create_chunks(text)
 
            st.info(
                f"Created {len(chunks)} chunks."
            )
 
            with st.spinner(
                "Creating Gemini embeddings and storing in ChromaDB..."
            ):
 
                number_of_chunks = store_chunks(chunks)
 
            st.success(
                f"PDF processed successfully! "
                f"{number_of_chunks} chunks stored."
            )
 
            st.session_state.pdf_processed = True
 
 
# --------------------------------------------------
# QUESTION SECTION
# --------------------------------------------------
 
if st.session_state.get("pdf_processed", False):
 
    st.divider()
 
    st.subheader("💬 Ask a question")
 
    question = st.text_input(
        "Enter your question:"
    )
 
    if st.button("🔍 Get Answer"):
 
        if not question.strip():
 
            st.warning("Please enter a question.")
 
        else:
 
            with st.spinner("Searching document..."):
 
                # Create embedding for question
                question_embedding = create_embedding(
                    question
                )
 
                # Search ChromaDB
                results = collection.query(
                    query_embeddings=[question_embedding],
                    n_results=3
                )
 
            documents = results["documents"][0]
 
            context = "\n\n".join(documents)
 
            st.subheader("📖 Retrieved Context")
 
            with st.expander("View retrieved information"):
 
                for i, doc in enumerate(documents):
 
                    st.write(
                        f"### Chunk {i + 1}"
                    )
 
                    st.write(doc)
 
            with st.spinner("Gemini is generating the answer..."):
 
                answer = generate_answer(
                    question,
                    context
                )
 
            st.subheader("🤖 Answer")
 
            st.write(answer)
 