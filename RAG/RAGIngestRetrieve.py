from langchain.embeddings import HuggingFaceEmbeddings  # to get embeddings
from langchain.vectorstores import Qdrant  # vector database
from langchain.docstore.document import Document
from langchain_qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM


class RAG:
    def __init__(self, collection_name, qdrant_url, embedding_model, llama_model):
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model,
                                           model_kwargs={'device': 'cpu'})
        self.llm = OllamaLLM(model=llama_model)

    def create_vector_qdrant(self, text_array):

        doc_array = [Document(page_content=txt, metadata={"source": "webpage1", "date": "2024-01-01"}) for txt in text_array]
        # embedding choice here is all-MiniLM-L6-v2, based on your hardware you can choose smaller size one or bigger size one.
        # embedding will help you to create vector space out of your text
        # embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
        #                                    model_kwargs={'device': 'cpu'})


        # client.delete_collection(collection_name=collection_name)  # if document exist delete it

        qdrant = Qdrant.from_documents(
            doc_array,
            self.embeddings,
            url=self.qdrant_url,
            collection_name=self.collection_name
        )
        print("Inserted")

    def answer_question(self, question):
        # embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
        #                                    model_kwargs={'device': 'cpu'})

        qDrant_vector = QdrantVectorStore.from_existing_collection(collection_name=self.collection_name, url=self.qdrant_url,
                                                                   embedding=self.embeddings)
        retriever = qDrant_vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        # llm = OllamaLLM(model="llama3.2:1b")

        prompt = """
        1. Use the following pieces of context to answer the question at the end.
        2. If you don't know the answer, just say that "Cannot answer the question from the context!" but don't make up an answer on your own.\n
        3. If the answer is found, Keep the answer crisp and limited to 3,4 sentences.

        Context: {context}

        Question: {question}

        Helpful Answer:"""

        QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

        llm_chain = LLMChain(
            llm=self.llm,
            prompt=QA_CHAIN_PROMPT,
            callbacks=None,
            verbose=True)

        document_prompt = PromptTemplate(
            input_variables=["page_content", "source"],
            template="Context:\ncontent:{page_content}\nsource:{source}",
        )

        combine_documents_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context",
            document_prompt=document_prompt,
            callbacks=None,
        )

        qa = RetrievalQA(
            combine_documents_chain=combine_documents_chain,
            verbose=True,
            retriever=retriever,
            return_source_documents=True,
        )

        res = qa(question)
        return res["result"]

    def extract_and_generate(self, context):
        # a custom prompt help us to assist our agent with better answer and make sure to not make up answers
        # custom_prompt_template = """Use the following pieces of information to answer the user’s question.
        # If you don’t know the answer, just say that you don’t know, don’t try to make up an answer.
        #
        # Context: {context}
        # Question: {question}
        #
        # Only return the helpful answer below and nothing else.
        # Helpful and Caring answer:
        # """
        #
        # prompt = PromptTemplate(template=custom_prompt_template,
        #                         input_variables=['context', 'question'])

        # llm = OllamaLLM(model="llama3.2:1b")

        # template = "What is a good name for a company that makes {product}?"
        template = """Summarize the context below 
                    Context: {context}
                    """

        prompt = PromptTemplate.from_template(template)

        llm_chain = LLMChain(prompt=prompt, llm=self.llm)

        # generated = llm_chain.run(product="mechanical keyboard")

        generated = llm_chain.run(context=context)
        return generated

if __name__ == "__main__":
    text_array = ['I am from West Bengal U where?',
             'Brother my house is Kolkata where is your house ??I am watching the Burj Khalifa',
             'No this voice like Atatrk', 'quot welcome to Kolkataquot this voice original creator link',
             'amar bari howrah',
             'Abai bhai tu jaanta bhi hai kitna bda hai ... wo to news waalo nai dikha diya diya to itna importance mil rha hai ... tum bhi jyada mt doo ... Last year ka kya rha wo nhi dikhaya ja rha dikhay dikhaya dikhaya dikhaya dikhaya dikhaya dikhaya dikhaya dikhaya dikhaya dikhaya dikhaya dikhaya dikhaya dikhaya dikhaya dikhaya dikhaya jaKolkata huh',
             'Aajjubhai note face', 'Stop still', 'Burj Khalifa is also coming to our Gopalganj district of Bihar',
             '2 biggest city in India', 'It is Bengali Power I am proud to be Bangla', '2023 pandal l']
    qdrant_url = "http://180.188.226.161:6333"
    collection_name = "test_collection_doc_6"
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    llama_model = "llama3.2:1b"
    rag = RAG(collection_name=collection_name, qdrant_url=qdrant_url, embedding_model=embedding_model, llama_model = llama_model)
