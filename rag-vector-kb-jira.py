INFO:chromadb.telemetry.product.posthog:Anonymized telemetry enabled. See                     https://docs.trychroma.com/telemetry for more information.
INFO:httpx:HTTP Request: POST http://127.0.0.1:11434/api/embed "HTTP/1.1 200 OK"
INFO:__main__:✅ Vectorstore created successfully.
🔧 RAG Troubleshooter

👤 Select error category (kubernetes/aws/network/other): getting 400 while building application
👤 Describe your error message: unable to tranfer artifacts

🤖 Please answer the following questions:
 - Please describe the issue in more detail.
 - What steps reproduce the error?
 - Have you tried restarting the service or checking logs?

👤 Your answers (summarize briefly): artifacts publishing failing

🧩 Searching knowledge base for possible solutions...
INFO:httpx:HTTP Request: POST http://127.0.0.1:11434/api/embed "HTTP/1.1 200 OK"
INFO:__main__:🔍 Best match distance: 0.995
INFO:__main__:📊 Match Confidence: 0.52%
INFO:__main__:⚠️ Low similarity — treating as no suitable match.

⚠️ No suitable solution found in RAG knowledge base.
📋 Creating Jira ticket for human review...
INFO:__main__:🎫 Jira ticket created successfully: KAN-6
✅ Jira Ticket Created: KAN-6
(.venv) root@ip-172-31-18-195:~# ls
'=0.3,'   myenv   rag-vector-kb-jira.py   rag-vector-kb.py   rag-vector-llm-kb.py   snap   test.py   venv
(.venv) root@ip-172-31-18-195:~#  cat  rag-vector-kb-jira.py
import logging
import requests
import base64
import json
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ------------------- Logging -------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------- Step 1: RAG Setup -------------------
def create_vectorstore(docs):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(docs)
    store = Chroma.from_documents(texts, embedding=embeddings)
    logger.info("✅ Vectorstore created successfully.")
    return store


# ------------------- Step 2: Clarifying Questions -------------------
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


# ------------------- Step 3: RAG Retrieval -------------------
def retrieve_solution(error_text, user_inputs, vectorstore, threshold=0.75):
    """Retrieve best matching doc and check similarity score."""
    query_context = f"Error: {error_text}\nUser details: {user_inputs}"
    try:
        results = vectorstore.similarity_search_with_score(query_context, k=3)
        if not results:
            return None

        best_doc, best_score = results[0]
        logger.info(f"🔍 Best match distance: {best_score:.3f}")

        confidence = (1 - best_score) * 100
        logger.info(f"📊 Match Confidence: {confidence:.2f}%")

        if best_score > threshold:  # high distance = poor match
            logger.info("⚠️ Low similarity — treating as no suitable match.")
            return None

        return best_doc.page_content
    except Exception as e:
        logger.error(f"❌ Retrieval failed: {e}")
        return None


# ------------------- Step 4: Jira Ticket Creation -------------------
def make_adf_description(text):
    """Convert plain text into Atlassian Document Format (ADF)."""
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": text}]}
        ],
    }

def create_jira_ticket(summary, description):
    jira_url = "https://leelapraveen19.atlassian.net"
    jira_user = "leelapraveen19@gmail.com"       # your Atlassian email
    jira_token = "ATATT3xFfGF0rQOE6c4vmTygCVEvEPdviu6US72NvmAKpqS4Ogd_VlTjdJgGWQxeET9gAc7qmlSmmsy18utJfMEUlc7acdn9LalvpReOAa3PQVOMRotwf3kAP41ODkswlEcoACpA0OHHFy6Jcm20vL6bYME3KE4hqsqaNEIVUhhNQIAHfqQAsY4=6596F8DC"
    jira_project_key = "KAN"                   # verified project key

    auth_str = f"{jira_user}:{jira_token}"
    auth_header = base64.b64encode(auth_str.encode()).decode()

    headers = {
        "Authorization": f"Basic {auth_header}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    payload = {
        "fields": {
            "project": {"key": jira_project_key},
            "summary": summary,
            "description": make_adf_description(description),
            "issuetype": {"name": "Task"}
        }
    }

    response = requests.post(f"{jira_url}/rest/api/3/issue", headers=headers, json=payload)

    if response.status_code == 201:
        issue_key = response.json().get("key")
        logger.info(f"🎫 Jira ticket created successfully: {issue_key}")
        return issue_key
    else:
        logger.error(f"❌ Jira creation failed: {response.status_code}\n{response.text}")
        return None


# ------------------- Step 5: Main Workflow -------------------
if __name__ == "__main__":
    docs = [
        Document(page_content="Kubernetes CrashLoopBackOff occurs when containers keep failing repeatedly due to bad config or permission issues."),
        Document(page_content="AWS AccessDenied errors usually mean IAM policy missing permissions like s3:GetObject or sts:AssumeRole."),
        Document(page_content="Network ConnectionRefusedError means the service on target port is not listening or blocked by firewall/NACL."),
    ]

    vectorstore = create_vectorstore(docs)

    print("🔧 RAG Troubleshooter\n")

    category = input("👤 Select error category (kubernetes/aws/network/other): ").strip().lower()
    error_text = input("👤 Describe your error message: ").strip()
    questions = ask_clarifying_questions(category)
    user_inputs = input("\n👤 Your answers (summarize briefly): ")

    print("\n🧩 Searching knowledge base for possible solutions...")
    solution = retrieve_solution(error_text, user_inputs, vectorstore)

    if solution:
        print("\n✅ Suggested Solution:\n", solution)
    else:
        print("\n⚠️ No suitable solution found in RAG knowledge base.")
        print("📋 Creating Jira ticket for human review...")

        description = (
            f"Category: {category}\n"
            f"Error: {error_text}\n"
            f"User Inputs: {user_inputs}\n"
            f"Status: RAG could not find a suitable fix."
        )
        summary = f"Unresolved {category.title()} Issue: {error_text[:80]}"

        issue_key = create_jira_ticket(summary, description)

        if issue_key:
            print(f"✅ Jira Ticket Created: {issue_key}")
        else:
            print("❌ Failed to create Jira ticket.")
