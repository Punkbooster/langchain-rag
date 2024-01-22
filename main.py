from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

if __name__ == "__main__":
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

  # initialize Embeddings. Embeding model accepts text and transforms it as a vector.
  embeddings = OllamaEmbeddings()

  qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    location=":memory:",
    collection_name="my_documents",
  )

  # connect to local ollama
  llm = Ollama(base_url="http://localhost:11434", model="llama2")

  # create a chain with Ollama embedded LLm
  # chain type stuff means that the context for LLm we are going to provide by ourselves from vector store
  # retriever is being used from a vector store
  qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=qdrant.as_retriever()
  )

  query = "What is a Vector DB? Give me a 15 word answer."

  result = qa({ "query": query })

  print(result)
