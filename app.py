import streamlit as st
from rag_pipeline import build_index, retrieve, generate_answer
from data.college_data import college_documents

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="College Student Assistant",
    page_icon="🎓",
    layout="centered"
)

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #EAF4FF, #F7F9FC);
}

/* Main title */
.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: 800;
    background: linear-gradient(90deg, #12355B, #2E6FBB);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #5B677A !important;
    font-size: 17px;
    margin-bottom: 28px;
}

/* Topic pill grid */
.topic-grid {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 10px;
    margin-bottom: 25px;
}

.topic-pill {
    background: white;
    border: 1px solid #D9E5F2;
    border-radius: 20px;
    padding: 8px 16px;
    font-size: 14px;
    font-weight: 600;
    color: #12355B;
    box-shadow: 0 2px 6px rgba(18,53,91,0.06);
}

/* Info box heading */
.info-heading {
    text-align: center;
    font-weight: 700;
    color: #12355B;
    margin-bottom: 12px;
    font-size: 16px;
}

/* Chat message */
.stChatMessage {
    border-radius: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    color: black !important;
    background-color: white !important;
    border-radius: 12px !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #666666 !important;
    opacity: 1 !important;
}

.stChatMessage p {
    color: #12355B !important;
}

.stMarkdown {
    color: black !important;
}

/* Clear chat button */
button[kind="secondary"] {
    border-radius: 20px !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD RAG INDEX ----------------

@st.cache_resource
def load_index():
    return build_index(college_documents)

indexed_docs = load_index()

# ---------------- HEADER ----------------

st.markdown(
    '<div class="main-title">🎓 College Student Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Your AI assistant for college information</div>',
    unsafe_allow_html=True
)

# ---------------- INFO BOX (TOPIC PILLS) ----------------

st.markdown('<div class="info-heading">💡 You can ask about</div>', unsafe_allow_html=True)

st.markdown("""
<div class="topic-grid">
    <div class="topic-pill">📚 Syllabus & Subjects</div>
    <div class="topic-pill">📝 Exams & Attendance</div>
    <div class="topic-pill">📢 College Notices</div>
    <div class="topic-pill">🎫 Hall Tickets</div>
    <div class="topic-pill">🏢 Student Services</div>
    <div class="topic-pill">💼 Placements</div>
</div>
""", unsafe_allow_html=True)

# ---------------- INITIALIZE CHAT ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- CLEAR CHAT ----------------

if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# ---------------- CHAT HISTORY ----------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])

# ---------------- CHAT INPUT ----------------

query = st.chat_input(
    "💬 Ask your college question..."
)

# ---------------- PROCESS QUESTION ----------------

if query:

    # Save user question
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # Display user question
    with st.chat_message("user"):
        st.write(query)

    # Generate AI answer
    with st.chat_message("assistant"):

        with st.spinner("🤖 Finding the best answer..."):

            results = retrieve(
                query,
                indexed_docs,
                top_k=2
            )

            answer = generate_answer(
                query,
                results
            )

        st.write(answer)

    # Save AI answer
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })