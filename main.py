import os

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import pinecone

from langchain.llms import Ollama

pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment="gcp-starter")

if __name__ == "__main__":
  print("hello Vector")

  # initialize the loader
  loader = TextLoader(
    "/Users/punkbooster/Projects/langchain/midium-blogs/medium-vector-database.txt"
  )

  # load full document
  document = loader.load()

  # initialize text splitter
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

  # split the text
  texts = text_splitter.split_documents(document)
  # command to check for how many chunks the text is splitted
  # print(len(texts))

  # initialize Embeddings. Embeding model accepts text and transforms it as a vector.
  embeddings = OpenAIEmbeddings(
    openai_api_key=os.environ.get("OPENAI_API_KEY")
  )

  # few things we are doing here:
  # take the text chunks, use openai embeddings api to convert text chunks to vectors, take these vectors and store them in pinecone database
  # index name is database name index created from pinecone.io
  doc_search = Pinecone.from_documents(
    texts, embeddings, index_name="langchain-doc-index"
  )

  # set which llm model we want to use with temperature.
  # temperature decides how creative the llm will be.
  llm = Ollama(base_url="http://localhost:11434", model="llama2")

  # create a chain with OpenAI LLm
  # chain type stuff means that the context for LLm we are going to provide by ourselves from vector store
  # retriever is being used from a vector store
  qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=doc_search.as_retriever()
  )

  query = "What is a Vector DB? Give me a 15 word answer."

  result = qa({ "query": query })

  print(result)
