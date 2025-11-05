# pip install -U "langchain>=0.3,<0.4" "langchain-community>=0.3,<0.4" langchain-ollama chromadb

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import logging
from langchain.memory import ConversationBufferMemory

# --- Setup logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Step 1: Create Ollama LLM ---
def create_llm(model_name="llama3", temperature=0.2):
    try:
        llm = OllamaLLM(model=model_name, temperature=temperature)
        return llm
    except Exception as e:
        logger.error(f"❌ Error loading Ollama model: {e}")
        raise

# --- Step 2: Create RAG retriever ---
def create_vectorstore(docs):
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = splitter.split_documents(docs)
        store = Chroma.from_documents(texts, embedding=embeddings)
        return store
    except Exception as e:
        logger.error(f"❌ Error creating vector store: {e}")
        raise

# --- Step 3: Build the new RAG Chain ---
def create_rag_agent(llm, vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    prompt = ChatPromptTemplate.from_template(
        "Answer the following question based on the provided context:\n\n{context}\n\nQuestion: {input}"
    )
    combine_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_chain)
    return rag_chain

# --- Step 4: Clarifying question generation ---
def get_clarifying_questions(error_text, llm):
    prompt = f"""
    The user reported this error or issue:

    "{error_text}"

    Generate 2-3 short and specific questions to better understand this problem before suggesting a fix.
    Return only the questions as a numbered list.
    """
    try:
        questions = llm.invoke(prompt)
        return questions
    except Exception as e:
        logger.error(f"❌ Error generating clarifying questions: {e}")
        return "Could you please provide more context about this error?"

# --- Step 5: Generate solution using RAG ---
def generate_solution(error_text, user_inputs, rag_chain):
    full_context = f"Error: {error_text}\nUser Inputs: {user_inputs}"
    try:
        response = rag_chain.invoke({"input": full_context})
        return response["answer"]
    except Exception as e:
        logger.error(f"❌ Error generating RAG response: {e}")
        return "Sorry, something went wrong while generating the solution."

# --- Example Flow ---
if __name__ == "__main__":
    docs = [
        Document(page_content="Error 'ConnectionRefusedError' often means a service is not running on the target port."),
        Document(page_content="In Kubernetes, 'CrashLoopBackOff' usually means the container failed to start repeatedly due to bad configuration or missing dependencies."),
        Document(page_content="In AWS, 'AccessDenied' errors usually mean the IAM role lacks permissions.")
    ]

    llm = create_llm("llama3")
    vectorstore = create_vectorstore(docs)
    rag_chain = create_rag_agent(llm, vectorstore)

    # --- Step 1: User reports an error ---
    user_error = input("👤 Enter your error message: ")

    # --- Step 2: Ask clarifying questions ---
    questions = get_clarifying_questions(user_error, llm)
    print("\n🤖 To help me solve this, please answer:")
    print(questions)

    # --- Step 3: Collect user answers ---
    user_inputs = input("\n👤 Your answers: ")

    # --- Step 4: Generate solution ---
    print("\n🧩 Searching knowledge base for solutions...")
    solution = generate_solution(user_error, user_inputs, rag_chain)
    print("\n✅ Suggested Solution:\n", solution)
