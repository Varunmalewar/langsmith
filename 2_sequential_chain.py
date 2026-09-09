from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from pydantic  import SecretStr

os.environ['LANGCHAIN_PROJECT'] = 'Sequential LLM App'

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash-lite",
    api_key = SecretStr(os.environ["GOOGLE_API_KEY"]),
    tempreature = 0.7 
)

parser = StrOutputParser()

chain = prompt1 | llm | parser | prompt2 | llm | parser

config = {
    'run_name' :'sequential_chain',
    'tags' : ['llm app','report generation','summarization'],
    'metadata': { 'model': 'gemini-3.5-flash-lite' ,'parser': 'StrOutputParser'}
}

result = chain.invoke({'topic': 'Unemployment in India'}, config=config)

print(result)
