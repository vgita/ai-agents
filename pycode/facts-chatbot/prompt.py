from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from redundant_filter_retriever import RedundantFilterRetriever
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI()

embeddings = OpenAIEmbeddings()
db = Chroma(
  persist_directory="facts-chatbot/data/emb",
  embedding_function=embeddings
)

retriever = RedundantFilterRetriever(embeddings = embeddings,chroma = db)

chain = RetrievalQA.from_chain_type(
  llm = chat,
  retriever = retriever,
  chain_type= "stuff"
)

result = chain.run("What is an interesting fact about the English language?")

print(result)