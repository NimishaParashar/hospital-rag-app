import sys
import os

# Fix for SQLite version issues with Chroma on certain OS distributions
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is missing.")

embedding_function = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=google_api_key
)

# Relative path to chroma_data directory inside backend/
chroma_dir = os.path.join(os.path.dirname(__file__), "chroma_data")
reviews_vector_db = Chroma(persist_directory=chroma_dir, embedding_function=embedding_function)

review_template_str = """Your job is to use patient reviews to answer questions about their experience at a hospital.
Use the following context to answer questions.
Be as detailed as possible, but don't make up any information that's not from the context.
If you don't know an answer, say you don't know.

{context}"""

review_system_prompt = SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=["context"], template=review_template_str))
review_human_prompt = HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=["question"], template="{question}"))
review_prompt_template = ChatPromptTemplate(input_variables=["context", "question"], messages=[review_system_prompt, review_human_prompt])

chat_model = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0, google_api_key=google_api_key)
reviews_retriever = reviews_vector_db.as_retriever(search_kwargs={"k": 10})

review_chain = (
    {"context": reviews_retriever, "question": RunnablePassthrough()}
    | review_prompt_template
    | chat_model
    | StrOutputParser()
)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = sys.argv[1]
        response = review_chain.invoke(query)
        print(response)