# ============================================================================
# BOT_ENGINE.PY - AI Chatbot Core Logic with RAG (Retrieval Augmented Generation)
# ============================================================================
# This module handles the main AI conversation logic, including:
# 1. Retrieving relevant PDF context when available (RAG)
# 2. Managing conversation history
# 3. Generating AI responses using Groq's LLM
# ============================================================================

# --- IMPORTS ---
import datetime  # For getting current date/time to provide temporal context to AI
from dotenv import load_dotenv  # Loads environment variables from .env file
import os  # Provides access to operating system environment variables
from langchain_groq import ChatGroq  # Groq's language model integration via LangChain
from langchain_community.vectorstores import FAISS  # Facebook AI Similarity Search - vector database

# --- ENVIRONMENT SETUP ---
load_dotenv(".env")  # Load all variables from .env file into environment
groq_api_key = os.getenv("GROQ_API_KEY")  # Extract Groq API key from environment variables


# ============================================================================
# MAIN CHAT FUNCTION
# ============================================================================
def Chat_with_Ai(prompt, history_list, rag_enabled, embedding_model):
    """
    Generate an AI response with optional PDF context retrieval (RAG).

    This is the brain of the chatbot. It:
    1. Searches PDF knowledge if available (RAG)
    2. Builds conversation context from history
    3. Adds current date/time for temporal awareness
    4. Sends everything to Groq's LLM
    5. Returns the AI's response

    Parameters:
    -----------
    prompt : str
        The current question/message from the user
        Example: "What is machine learning?"

    history_list : list of dict
        Previous conversation messages in format:
        [{"role": "user", "message": "Hi"}, {"role": "assistant", "message": "Hello!"}]
        This gives the AI context about what was discussed before

    rag_enabled : bool
        Whether PDF knowledge base is loaded
        True = Use PDF context in answers
        False = Answer from general knowledge only

    embedding_model : HuggingFaceEmbeddings
        The model that converts text into numerical vectors
        Used for finding similar PDF chunks to the user's question

    Returns:
    --------
    str
        The AI's text response to the user's question
    """

    # Initialize empty variable to store PDF-related information
    pdf_context = ""

    # ========================================================================
    # STEP 1: RAG (Retrieval Augmented Generation)
    # ========================================================================
    # If a PDF has been uploaded and indexed, retrieve relevant sections
    if rag_enabled:
        try:
            # --- Load the FAISS Vector Database ---
            # FAISS stores embeddings (numerical representations) of PDF chunks
            # These embeddings allow us to find semantically similar text
            loaded_database = FAISS.load_local(
                "faiss_index",  # Directory name where index was saved
                embedding_model,  # Same model used to create the embeddings
                allow_dangerous_deserialization=True  # Needed because FAISS uses pickle (serialization)
                # Note: Only use with trusted data sources
            )

            # --- Semantic Search ---
            # Find the top 3 most relevant chunks from the PDF
            # How it works:
            # 1. Convert user's question to an embedding (vector)
            # 2. Compare this vector to all PDF chunk vectors
            # 3. Return the chunks with highest similarity scores
            results = loaded_database.similarity_search(
                prompt,  # User's question
                k=3  # Number of results to return (top 3 most similar chunks)
            )

            # --- Combine Retrieved Chunks ---
            # Join the top 3 chunks with double newlines for readability
            # result.page_content contains the actual text from each chunk
            pdf_context = "\n\n".join([result.page_content for result in results])

        except Exception as e:
            # If FAISS loading fails (e.g., no PDF uploaded yet, corrupted index)
            # Print error for debugging but continue without PDF context
            print(f"Error loading FAISS: {e}")
            # pdf_context remains empty string, so AI will answer without PDF knowledge

    # ========================================================================
    # STEP 2: BUILD CONVERSATION HISTORY
    # ========================================================================
    # Include recent conversation for context-aware responses
    context_text = ""

    # Only use last 6 messages to:
    # 1. Stay within token limits (LLMs have max input size)
    # 2. Keep context relevant (very old messages may not matter)
    # 3. Reduce processing time and cost
    for chat in history_list[-6:]:  # Slice gets last 6 items from list

        # Format user messages
        if chat["role"] == "user":
            context_text += f"User: {chat['message']}\n"

        # Format assistant (AI) messages
        else:
            context_text += f"Assistant: {chat['message']}\n"

    # Result example:
    # User: What is AI?
    # Assistant: AI is artificial intelligence...
    # User: Give me examples
    # Assistant: Examples include ChatGPT, Siri...

    # ========================================================================
    # STEP 3: ADD TEMPORAL CONTEXT
    # ========================================================================
    # Give the AI awareness of current date/time
    # This helps with questions like "What day is it?" or time-sensitive queries

    # datetime.datetime.now() - Gets current date and time
    # strftime() - Formats the datetime into a readable string
    current_date = datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

    # Format breakdown:
    # %A = Full weekday name (Monday, Tuesday, etc.)
    # %B = Full month name (January, February, etc.)
    # %d = Day of month (01-31)
    # %Y = Full year (2026)
    # %I = Hour in 12-hour format (01-12)
    # %M = Minute (00-59)
    # %p = AM/PM

    # Example output: "Monday, February 03, 2026 at 02:30 PM"

    # ========================================================================
    # STEP 4: CONSTRUCT FINAL PROMPT
    # ========================================================================
    # Build the complete prompt that will be sent to the AI

    if pdf_context:
        # --- PROMPT WITH PDF KNOWLEDGE (RAG Mode) ---
        # This version includes retrieved PDF chunks for grounded answers
        final_prompt = (
            f"Current Date: {current_date}. "  # Temporal awareness
            f"You are a helpful AI assistant. Answer the user's question based on the following PDF context:\n\n"
            f"--- PDF CONTEXT ---\n{pdf_context}\n--- END CONTEXT ---\n\n"  # Retrieved chunks
            f"Conversation History:\n{context_text}\n"  # Previous messages
            f"User: {prompt}\n"  # Current question
            f"Assistant:"  # Prompt for AI to continue
        )
    else:
        # --- PROMPT WITHOUT PDF (Normal Chat Mode) ---
        # This version relies only on the AI's training knowledge
        final_prompt = (
            f"Current Date: {current_date}. "  # Temporal awareness
            f"You are a helpful AI assistant.\n\n"
            f"Conversation History:\n{context_text}\n"  # Previous messages
            f"User: {prompt}\n"  # Current question
            f"Assistant:"  # Prompt for AI to continue
        )

    # ========================================================================
    # STEP 5: INITIALIZE GROQ LLM (Language Model)
    # ========================================================================
    llm = ChatGroq(
        # --- Model Selection ---
        model="llama-3.3-70b-versatile",
        # This is Meta's Llama 3.3 model with 70 billion parameters
        # "versatile" variant means it's good at various tasks
        # Other options: "llama-3.1-8b-instant" (faster, less capable)

        # --- API Authentication ---
        api_key=groq_api_key,
        # Your unique key that identifies and authorizes you with Groq

        # --- Temperature Parameter ---
        temperature=0.7,
        # Controls randomness/creativity of responses
        # 0.0 = Deterministic, same answer every time (good for factual Q&A)
        # 0.7 = Balanced creativity and consistency (RECOMMENDED)
        # 1.0+ = Very creative but potentially inconsistent

        # --- Token Limit ---
        max_tokens=1024
        # Maximum length of AI's response
        # 1 token ≈ 0.75 words (rough estimate)
        # 1024 tokens ≈ 750-800 words
        # Limits cost and ensures responses aren't too long
    )

    # ========================================================================
    # STEP 6: GET AI RESPONSE
    # ========================================================================
    # Send the complete prompt to Groq's API and receive response
    response = llm.invoke(final_prompt)

    # invoke() does:
    # 1. Sends final_prompt to Groq's servers
    # 2. Groq processes it with the Llama model
    # 3. Returns a response object

    # The response object contains:
    # - response.content: The actual text response
    # - response.response_metadata: Info about tokens used, model, etc.

    # ========================================================================
    # STEP 7: RETURN RESPONSE
    # ========================================================================
    # Extract only the text content from the response object
    return response.content
    # This is what gets displayed to the user in the chat interface

