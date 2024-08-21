import dotenv
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

Reviews_csv_path = "data/reviews.csv"
Reviews_chroma_path = "chroma_data"

dotenv.load_dotenv()

loader = CSVLoader(file_path=Reviews_csv_path, source_column="review")
reviews = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size =1000, chunk_overlap =200)
splits = text_splitter.split_documents(reviews)

reviews_vector_db = Chroma.from_documents(
    documents=splits, embedding= OpenAIEmbeddings(), persist_directory=Reviews_chroma_path
)

