from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import logging
from langchain.memory import ConversationBufferMemory


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Step 1: Setup RAG Components ---
def create_vectorstore(docs):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(docs)
    store = Chroma.from_documents(texts, embedding=embeddings)
    logger.info("✅ Vectorstore created successfully.")
    return store


def create_rag_retriever(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return retriever


# --- Step 2: Predefined clarifying questions ---
ERROR_CLARIFYING_QUESTIONS = {
    "kubernetes": [
        "What is the exact error message or pod status?",
        "Can you share the container image name?",
        "What do you see in `kubectl logs <pod>`?",
    ],
    "aws": [
        "Which AWS service is showing the error?",
        "Can you share the IAM role or user involved?",
        "Did this error appear during deployment or runtime?",
    ],
    "network": [
        "Which port or protocol is being used?",
        "Is there a firewall or NACL rule in between?",
        "Can you ping or telnet to the target host?",
    ],
}

def ask_clarifying_questions(category):
    questions = ERROR_CLARIFYING_QUESTIONS.get(category.lower(), [
        "Please describe the issue in more detail.",
        "What steps reproduce the error?",
        "Have you tried restarting the service or checking logs?",
    ])
    print("\n🤖 Please answer the following questions:")
    for q in questions:
        print(" -", q)
    return questions


# --- Step 3: Combine inputs before RAG retrieval ---
def retrieve_solution(error_text, user_inputs, retriever):
    query_context = f"Error: {error_text}\nUser details: {user_inputs}"
    try:
        results = retriever.get_relevant_documents(query_context)
        if not results:
            return "No matching solutions found in the knowledge base."
        best_doc = results[0]
        return best_doc.page_content
    except Exception as e:
        logger.error(f"❌ Retrieval failed: {e}")
        return "An error occurred while searching for solutions."


# --- Example Run ---
if __name__ == "__main__":
    # Step 1: Load some example knowledge base documents
    docs = [
        Document(page_content="Kubernetes CrashLoopBackOff occurs when containers keep failing repeatedly due to bad config or permission issues."),
        Document(page_content="AWS AccessDenied errors usually mean IAM policy missing permissions like s3:GetObject or sts:AssumeRole."),
        Document(page_content="Network ConnectionRefusedError means the service on target port is not listening or blocked by firewall/NACL."),
    ]

    # Step 2: Setup vector store
    vectorstore = create_vectorstore(docs)
    retriever = create_rag_retriever(vectorstore)

    # Step 3: User interaction
    print("🔧 RAG Troubleshooter\n")

    category = input("👤 Select error category (kubernetes/aws/network/other): ").strip().lower()
    error_text = input("👤 Describe your error message: ").strip()

    # Ask clarifying questions (no LLM)
    questions = ask_clarifying_questions(category)
    user_inputs = input("\n👤 Your answers (summarize briefly): ")

    # Step 4: Retrieve best-matched solution
    print("\n🧩 Searching knowledge base for possible solutions...")
    solution = retrieve_solution(error_text, user_inputs, retriever)
    print("\n✅ Suggested Solution:\n", solution)
