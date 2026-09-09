# pip install -U langchain langchain-openai langchain-community faiss-cpu pypdf python-dotenv langsmith

import os
from dotenv import load_dotenv

from langsmith import traceable  # <-- key import

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI ,GoogleGenerativeAIEmbeddings
import os 
from pydantic import SecretStr
import time 

# --- LangSmith env (make sure these are set) ---
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=...
# LANGCHAIN_PROJECT=pdf_rag_demo

load_dotenv()

os.environ['LANGCHAIN_PROJECT'] = 'Rag LLM App V1'


PDF_PATH = "D:\\Langsmith\\langsmith-masterclass\\islr.pdf"  # change to your file
MAX_PAGES = 10  # only load + chunk the first N pages of the PDF
class RateLimitedEmbeddings(GoogleGenerativeAIEmbeddings):
    """Embeds in batches with a pause between batches and automatic
    retry-with-backoff when the free-tier 429 quota error is hit."""

    def embed_documents(self, texts, batch_size=100):
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            if i > 0:  # stay under the 100 requests/minute free-tier limit
                print(f"  ...pausing 60s to respect free-tier rate limit ({i}/{len(texts)} chunks done)")
                time.sleep(60)
            for attempt in range(5):
                try:
                    results.extend(super().embed_documents(batch, batch_size=batch_size))
                    break
                except Exception as e:
                    if "RESOURCE_EXHAUSTED" in str(e) and attempt < 4:
                        wait = 60 * (attempt + 1)
                        print(f"  429 quota hit, waiting {wait}s before retry...")
                        time.sleep(wait)
                    else:
                        raise
        return results

# ---------- traced setup steps ----------
@traceable(name="load_pdf")
def load_pdf(path: str):
    loader = PyPDFLoader(path)
    docs = loader.load()  # list[Document] (one Document per page)
    docs = docs[:MAX_PAGES]  # keep only the first 10 pages
    print(f"Loaded {len(docs)} pages from {path}")
    return docs

@traceable(name="split_documents")
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

@traceable(name="build_vectorstore")
def build_vectorstore(splits):
    emb = RateLimitedEmbeddings(model="models/gemini-embedding-001")
    # FAISS.from_documents internally calls the embedding model:
    vs = FAISS.from_documents(splits, emb)
    return vs

# You can also trace a “setup” umbrella span if you want:
@traceable(name="setup_pipeline")
def setup_pipeline(pdf_path: str):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs)
    vs = build_vectorstore(splits)
    return vs

# ---------- pipeline ----------
llm = ChatGoogleGenerativeAI(
    model = "gemini-3.1-flash-lite",
    api_key = SecretStr(os.environ["GOOGLE_API_KEY"]),
    temperature = 0.7
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# Build the index under traced setup
# Cache the index so repeated runs don't re-embed and burn free-tier quota
INDEX_DIR = "faiss_index_v2_first10pages"
emb_model = "models/gemini-embedding-001"

if os.path.exists(INDEX_DIR):
    print("Loading cached FAISS index...")
    _emb = RateLimitedEmbeddings(model=emb_model)
    vectorstore = FAISS.load_local(INDEX_DIR, _emb, allow_dangerous_deserialization=True)
else:
    vectorstore = setup_pipeline(PDF_PATH)
    vectorstore.save_local(INDEX_DIR)
    print("Index cached to", INDEX_DIR)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
})

chain = parallel | prompt | llm | StrOutputParser()

# ---------- run a query (also traced) ----------
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ").strip()

# Give the visible run name + tags/metadata so it’s easy to find:
config = {
    "run_name": "pdf_rag_query"
}

ans = chain.invoke(q, config=config)
print("\nA:", ans)
