# Import necessary libraries
import streamlit as st  # Streamlit for building web UI
import pdf_handler  # Custom module for PDF processing
import bot_engine  # Custom module for AI chat logic
from langchain_huggingface import HuggingFaceEmbeddings  # For text embeddings


# --- EMBEDDING MODEL LOADER ---
@st.cache_resource(scope="session")  # Cache the model so it loads only once per session
def load_embedding_model():
    """
    Load the sentence transformer model for creating embeddings.
    This model converts text into numerical vectors for similarity search.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Load the embedding model once
embedding_model = load_embedding_model()

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Verma AI",  # Browser tab title
    page_icon="🤖",  # Browser tab icon
    layout="wide"  # Use full screen width like ChatGPT
)

# --- 2. CUSTOM CSS STYLING ---
# Hide Streamlit's default menu and footer for a cleaner look
st.markdown("""
<style>
    /* Hide the Streamlit main menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Add some padding to the bottom */
    .stChatMessage {
        padding-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. SESSION STATE INITIALIZATION ---
# Session state persists data across reruns (when user interacts with UI)

# Initialize chat history if it doesn't exist
if "history" not in st.session_state:
    st.session_state.history = []  # List of all messages (user + AI)

# Initialize RAG status (whether PDF is loaded)
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = False  # False by default (no PDF loaded)

# --- 4. SIDEBAR (Control Panel) ---
with st.sidebar:
    # Sidebar title
    st.title("📂 Knowledge Base")
    st.write("Upload your PDF to train the brain.")

    # --- PDF UPLOAD SECTION ---
    # File uploader widget (accepts only PDF files)
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    # Process button for the uploaded PDF
    if st.button("Submit & Process", type="primary"):  # "primary" makes it stand out
        if uploaded_file:
            # Show a spinner while processing
            with st.spinner("Reading & Indexing..."):
                try:
                    # Step 1: Extract text from PDF
                    raw_text = pdf_handler.get_pdf_text(uploaded_file)

                    # Step 2: Split text into smaller chunks
                    text_chunks = pdf_handler.get_text_chunks(raw_text)

                    # Step 3: Create embeddings and save to FAISS database
                    pdf_handler.get_vector_store(text_chunks, embedding_model)

                    # Step 4: Enable RAG mode
                    st.session_state.rag_enabled = True

                    # Show success message
                    st.success("Brain Updated! ✅")

                except Exception as e:
                    # If something goes wrong, show error
                    st.error(f"Error processing PDF: {str(e)}")
        else:
            # If no file uploaded, show warning
            st.warning("Please upload a file first.")

    # Separator line
    st.markdown("---")

    # --- CLEAR CONVERSATION BUTTON ---
    if st.button("🗑️ Clear Conversation"):
        # Reset chat history
        st.session_state.history = []
        # Rerun the app to refresh UI
        st.rerun()

    # --- DISPLAY RAG STATUS ---
    # Show whether PDF knowledge is active
    if st.session_state.rag_enabled:
        st.success("📚 PDF Knowledge Active")
    else:
        st.info("💬 Chat Mode")

# --- 5. MAIN CHAT INTERFACE ---

# Main title of the chatbot
st.subheader("🤖 Alok's Dynamic AI")

# --- DISPLAY CHAT HISTORY ---
# Loop through all previous messages and display them
for chat in st.session_state.history:
    # Each message is rendered in a chat bubble
    with st.chat_message(chat["role"]):  # "role" is either "user" or "assistant"
        st.markdown(chat["message"])  # Display the message text

# --- CHAT INPUT BOX ---
# The := operator assigns AND checks the value in one line
if prompt := st.chat_input("Ask me anything..."):

    # --- Step 1: Display User's Message ---
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Step 2: Save User's Message to History ---
    st.session_state.history.append({"role": "user", "message": prompt})

    # --- Step 3: Get AI Response ---
    with st.chat_message("assistant"):
        # Show "Thinking..." while waiting for response
        with st.spinner("Thinking..."):
            try:
                # Call the AI function with all necessary parameters
                response = bot_engine.Chat_with_Ai(
                    prompt,  # Current user question
                    st.session_state.history,  # Conversation history
                    st.session_state.rag_enabled,  # Whether to use PDF knowledge
                    embedding_model  # Embedding model for RAG
                )

                # Display the AI's response
                st.markdown(response)

                # Save AI's response to history
                st.session_state.history.append({"role": "assistant", "message": response})

            except Exception as e:
                # If something goes wrong, show error message
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                # Also save error to history
                st.session_state.history.append({"role": "assistant", "message": error_msg})