# pip install -U langchain langchain-openai langchain-community faiss-cpu pypdf python-dotenv

import os
import time
from dotenv import load_dotenv
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


load_dotenv()  # expects OPENAI_API_KEY in .env

os.environ['LANGCHAIN_PROJECT'] = 'Rag LLM App V1'

PDF_PATH = "D:\\Langsmith\\langsmith-masterclass\\islr.pdf"  # <-- change to your PDF filename

# 1) Load PDF
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()  # one Document per page

# 2) Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = splitter.split_documents(docs)

# 3) Embed + index
# text-embedding-004 has been retired by Google (404 NOT_FOUND).
# gemini-embedding-001 is the current recommended model (free tier: 100 RPM / 30k TPM).
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

emb = RateLimitedEmbeddings(model="models/gemini-embedding-001")

# Cache the index so repeated runs don't re-embed everything and burn quota again
INDEX_DIR = "faiss_index_islr"
if os.path.exists(INDEX_DIR):
    print("Loading cached FAISS index...")
    vs = FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)
else:
    print(f"Embedding {len(splits)} chunks (first run may take a while on the free tier)...")
    vs = FAISS.from_documents(splits, emb)
    vs.save_local(INDEX_DIR)
    print("Index cached to", INDEX_DIR)
retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# 4) Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# 5) Chain
llm = ChatGoogleGenerativeAI(
    model = "gemini-3.1-flash-lite",
    api_key = SecretStr(os.environ["GOOGLE_API_KEY"]),
    temperature = 0.7
)
def format_docs(docs): 
    return "\n\n".join(d.page_content for d in docs)

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()

# 6) Ask questions
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ")
ans = chain.invoke(q.strip())
print("\nA:", ans)
