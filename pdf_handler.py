# Import necessary libraries
import pypdf as py  # For reading PDF files
from langchain_text_splitters import RecursiveCharacterTextSplitter  # For splitting text into chunks
from langchain_community.vectorstores import FAISS  # Vector database for storing embeddings


def get_pdf_text(uploaded_file):
    """
    Extract all text from a PDF file.

    Args:
        uploaded_file: The uploaded PDF file object from Streamlit

    Returns:
        str: All text from the PDF combined into a single string
    """

    parts = []  # List to store text from each page

    # Create a PDF reader object
    reader = py.PdfReader(uploaded_file)

    # Loop through each page in the PDF
    for page in range(len(reader.pages)):
        # Get the page object
        text = reader.pages[page]

        # Extract text from the page and add to list
        parts.append(text.extract_text())

    # Join all page texts into one long string
    return ''.join(parts)


def get_text_chunks(text):
    """
    Split long text into smaller chunks for better processing.

    Why chunk?
    - AI models have token limits
    - Smaller chunks = more precise retrieval
    - Overlap ensures context isn't lost at boundaries

    Args:
        text (str): The long text to split

    Returns:
        list: List of text chunks
    """

    # Create a text splitter with specific parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Each chunk will be ~1000 characters
        chunk_overlap=200  # 200 characters overlap between chunks to maintain context
    )

    # Split the text and return chunks
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(chunks, embedding_model):
    """
    Convert text chunks into embeddings and store in FAISS vector database.

    How it works:
    1. Each text chunk is converted to a numerical vector (embedding)
    2. These vectors are stored in FAISS for fast similarity search
    3. When user asks a question, we find the most similar chunks

    Args:
        chunks (list): List of text chunks from the PDF
        embedding_model: The model to convert text to vectors

    Returns:
        FAISS: The created FAISS index
    """

    # Create FAISS index from text chunks
    # This converts each chunk to an embedding and stores it
    faiss_index = FAISS.from_texts(chunks, embedding_model)

    # Save the FAISS index to disk so we can load it later
    # This creates a folder called "faiss_index" with the database files
    faiss_index.save_local("faiss_index")

    return faiss_index