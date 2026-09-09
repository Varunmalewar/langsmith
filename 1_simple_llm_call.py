from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from pydantic  import SecretStr

load_dotenv()

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash-lite",
    api_key = SecretStr(os.environ["GOOGLE_API_KEY"]),
)
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | llm | parser

# Run it
result = chain.invoke({"question": "What is the capital of India?"})
print(result)
