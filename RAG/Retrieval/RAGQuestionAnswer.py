from langchain_qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.embeddings import HuggingFaceEmbeddings

qdrant_url = "http://180.188.226.161:6333"
collection_name = "test_collection_doc_6"


def answer_question(question):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    qDrant_vector = QdrantVectorStore.from_existing_collection(collection_name=collection_name, url=qdrant_url, embedding=embeddings)
    # qDrant_vector = QdrantVectorStore(collection_name=collection_name, url=qdrant_url, embedding=embeddings)
    retriever = qDrant_vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = OllamaLLM(model="llama3.2:1b")


    prompt = """
    1. Use the following pieces of context to answer the question at the end.
    2. If you don't know the answer, just say that "Cannot answer the question from the context!" but don't make up an answer on your own.\n
    3. If the answer is found, Keep the answer crisp and limited to 3,4 sentences.

    Context: {context}

    Question: {question}

    Helpful Answer:"""

    # prompt = """
    # 1. Use the following pieces of context to answer the question at the end.
    # 2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
    # 3. Keep the answer crisp and limited to 3,4 sentences.
    #
    # Context: {context}
    #
    # Question: {question}
    #
    # Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

    llm_chain = LLMChain(
        llm=llm,
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

    # res = qa("How encoding works?")["result"]
    res = qa(question)
    # res = qa("Give some examples of red colored fruit?")["result"]
    # res = qa("Give some examples of Red Fruit")["result"]
    return res["result"]

if __name__ == '__main__':
    # question = "What the document is about?"
    question = "I am from which location?"
    res = answer_question(question)
    print(res)