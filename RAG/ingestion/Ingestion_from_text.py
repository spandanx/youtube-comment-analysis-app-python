from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings  # to get embeddings
from langchain.vectorstores import Qdrant  # vector database
from qdrant_client import QdrantClient
from langchain.llms import CTransformers  # to get llm
from langchain.text_splitter import RecursiveCharacterTextSplitter  # splitting text into chunks
from langchain.chains import RetrievalQA  # building Retrieval chain
from langchain.document_loaders import PyPDFLoader, UnstructuredURLLoader  # to read pdfs, urls
# from langchain_community.llms import OpenLLM
from langchain_ollama import OllamaLLM
qdrant_url = "http://180.188.226.161:6333"
# llm_url = "http://localhost:11434"
# qdrant_api_key = ""
collection_name = "test_collection_5"

# def get_documents(url):
#     if ".pdf" in url:
#         loader = PyPDFLoader(url, extract_images=True)
#         pages = loader.load()
#     elif ".html" in url:
#         url = [url]
#         loader = UnstructuredURLLoader(urls=url)
#         pages = loader.load()
#     else:
#         return "unknown type for document_loaders"
#     return pages


def create_vector_qdrant():
    # get the documents in langchain format
    # loader = PyPDFLoader("../data/Retrieval-Augmented-Generation-for-NLP.pdf")
    # documents = loader.load()
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # texts = text_splitter.split_documents(documents)
    texts = ['I am from West Bengal U where?', 'Brother my house is Kolkata where is your house ??I am watching the Burj Khalifa', 'No this voice like Atatrk', 'quot welcome to Kolkataquot this voice original creator link', 'amar bari howrah', 'Abai bhai tu jaanta bhi hai kitna bda hai ... wo to news waalo nai dikha diya diya to itna importance mil rha hai ... tum bhi jyada mt doo ... Last year ka kya rha wo nhi dikhaya ja rha dikhay dikhaya dikhaya dikhaya dikhaya dikhaya dikhaya dikhaya dikhaya dikhaya dikhaya dikhaya dikhaya dikhaya dikhaya dikhaya dikhaya dikhaya jaKolkata huh', 'Aajjubhai note face', 'Stop still', 'Burj Khalifa is also coming to our Gopalganj district of Bihar', '2 biggest city in India', 'It is Bengali Power I am proud to be Bangla', '2023 pandal l']

    # embedding choice here is all-MiniLM-L6-v2, based on your hardware you can choose smaller size one or bigger size one.
    # embedding will help you to create vector space out of your text
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    # client.delete_collection(collection_name=collection_name)  # if document exist delete it

    qdrant = Qdrant.from_texts(
    # qdrant = Qdrant.from_documents(
        texts,
        embeddings,
        url=qdrant_url,
        collection_name=collection_name
    )
    print("Inserted")

if __name__ == "__main__":
    create_vector_qdrant()