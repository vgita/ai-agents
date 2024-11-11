from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

text_splitter = CharacterTextSplitter(
  separator="\n",
  chunk_size=200,
  chunk_overlap=0
)

loader = TextLoader("facts-chatbot/data/facts.txt")
docs = loader.load_and_split(text_splitter=text_splitter)

db = Chroma.from_documents(
  docs,
  embedding=embeddings,
  persist_directory="facts-chatbot/data/emb"
)

results = db.similarity_search_with_score(
  "What is an interesting fact about the English language?",
  k=4
)

for result in results:
  print("\n")
  print(result[0].page_content)