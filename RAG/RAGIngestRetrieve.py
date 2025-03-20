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
    def __init__(self, qdrant_url, embedding_model, llama_model):
        self.qdrant_url = qdrant_url
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model,
                                           model_kwargs={'device': 'cpu'})
        self.llm = OllamaLLM(model=llama_model)

    def ingest_vector(self, text_array, collection_name):

        doc_array = [Document(page_content=txt["text"], metadata=txt["metadata"]) for txt in text_array]
        # embedding choice here is all-MiniLM-L6-v2, based on your hardware you can choose smaller size one or bigger size one.

        qdrant = Qdrant.from_documents(
            doc_array,
            self.embeddings,
            url=self.qdrant_url,
            collection_name=collection_name,
            force_recreate=True
        )
        print("Inserted")

    def answer_question(self, question, context, collection_name):
        print("Called RAG Q&A", question, collection_name)
        qDrant_vector = QdrantVectorStore.from_existing_collection(collection_name=collection_name, url=self.qdrant_url,
                                                                   embedding=self.embeddings)
        retriever = qDrant_vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})


        prompt = """
        1. Use the following pieces of context to answer the question at the end.
        2. If you don't know the answer, just say that "Cannot answer the question!" but don't make up an answer on your own.\n
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
            input_variables=["page_content", "video_id"],
            template="Context:\ncontent:{page_content}\nsource:{video_id}",
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
        return {"answer": res["result"]}

    def summarizeText(self, context):

        # template = """Summarize the context below in 1,2 sentences
        #             Context: {context}
        #             """
        template = """
        1. Create a summary of the context provided below in 3,4 sentences.
        2. If you cannot summarize, just say that "Cannot summarize the statements!" but don't make up an answer on your own.\n
        3. If you find the summary, provide on the summary but no prompt sentences.
        Context = {context}
        """

        prompt = PromptTemplate.from_template(template)

        llm_chain = LLMChain(prompt=prompt, llm=self.llm)

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
    rag = RAG(qdrant_url=qdrant_url, embedding_model=embedding_model, llama_model = llama_model)
    texts = [
    {"text": 'I am from West Bengal U where?', "metadata":{'comment_id': 'UgxBMicJBMOynhGwioZ4AaABAg', 'reply_id': 'UgxBMicJBMOynhGwioZ4AaABAg.9TOYLFo88e29TS9FPoLcZI', 'type': 'reply', 'video_id': 'YPb3yfR2ssg'}},
    {"text":'No this voice like Atatrk', "metadata":{'comment_id': 'UgxBMicJBMOynhGwioZ4AaABAg', 'reply_id': 'UgxBMicJBMOynhGwioZ4AaABAg.9TOYLFo88e29TWSIWWfkhk', 'type': 'reply', 'video_id': 'YPb3yfR2ssg'}}
    ]
    texts1 = [{'text': 'It39s udaan Himalayan picture Not Ramada at no.5', 'metadata': {'video_id': '59YtVGokCAE', 'comment_id': 'Ugw5cIs929q5HCQrJyx4AaABAg', 'reply_id': '', 'type': 'comment'}},
              {'text': 'I39m proud to a Gorkha Darjeeling is my birth place', 'metadata': {'video_id': '59YtVGokCAE', 'comment_id': 'UgxA2T128IEsb_Ki5XJ4AaABAg', 'reply_id': '', 'type': 'comment'}},
              {'text': 'How much is the price of the foot day on the hotel', 'metadata': {'video_id': '59YtVGokCAE', 'comment_id': 'Ugw9-fAMn0fZoecEW-p4AaABAg', 'reply_id': '', 'type': 'comment'}},
              {'text': 'Please price to mention krte sare hotels ka', 'metadata': {'video_id': 'M-xc6DbpWr0', 'comment_id': 'UgzoZ5zsRnWeLPbhlI94AaABAg', 'reply_id': '', 'type': 'comment'}},
              {'text': 'Thank you for all the important information. But since you have stayed in so many hotels in Darjeeling, can you suggest which hotel will be ideal for me? Can you guide which hotels have a balcony with a great view within a tariff of 30004000 per night during the offseason? I don39t need to be on the main road, but I am looking for a hotel with great balcony sight where we can sit and enjoy Kanchanjunga. I39m planning to visit Darjeeling in July 2023.', 'metadata': {'video_id': 'M-xc6DbpWr0', 'comment_id': 'UgwYZgCDBtfaLsBbgSF4AaABAg', 'reply_id': '', 'type': 'comment'}},
              {'text': 'Love how most of these hotels are surrounded by beautiful views', 'metadata': {'video_id': 'M-xc6DbpWr0', 'comment_id': 'Ugyz2gOa2hu3SYh2Mnx4AaABAg', 'reply_id': '', 'type': 'comment'}},
              {'text': 'Very helpful video keep it up', 'metadata': {'video_id': 'M-xc6DbpWr0', 'comment_id': 'UgzONnlJ8ckSGln_t4h4AaABAg', 'reply_id': '', 'type': 'comment'}},
              {'text': 'Very nice. These ideas definitely help', 'metadata': {'video_id': 'M-xc6DbpWr0', 'comment_id': 'UgySk5F8KrvhQeX5R0N4AaABAg', 'reply_id': '', 'type': 'comment'}},
              {'text': 'Looking forward to experience such luxury stay...', 'metadata': {'video_id': 'M-xc6DbpWr0', 'comment_id': 'UgxEu8fAxftzceyupbV4AaABAg', 'reply_id': '', 'type': 'comment'}},
              {'text': 'Seriously!!! Stop misleading people and public who wish to come Darjeeling.', 'metadata': {'video_id': 'hlbzz7JTGrg', 'comment_id': 'Ugw8ORNF74Rc_CiZ6yN4AaABAg', 'reply_id': '', 'type': 'comment'}},
              {'text': 'you missed Elgin...one of the finest hotels!', 'metadata': {'video_id': 'hlbzz7JTGrg', 'comment_id': 'Ugxma6yNFsmmLShJOo54AaABAg', 'reply_id': '', 'type': 'comment'}},
              {'text': 'Do you smoke some sort weeds?? You video is about luxury hotels in Darjeeling and you have not included Mayfair, Sinclairs, Windermere, Sterling and Ramada! This video is an absolute joke', 'metadata': {'video_id': 'hlbzz7JTGrg', 'comment_id': 'Ugxz_K0YJDWdsHzehut4AaABAg', 'reply_id': '', 'type': 'comment'}},
              {'text': 'Very nice and informative', 'metadata': {'video_id': 'hlbzz7JTGrg', 'comment_id': 'UgzVyeg5OY0ala9aTpN4AaABAg', 'reply_id': '', 'type': 'comment'}},
              {'text': 'Please recommend a hotel that will have a Kanchanjanga view room near Mal.Bastard', 'metadata': {'video_id': 'hlbzz7JTGrg', 'comment_id': 'UgwTM_Bgb773uc-3XTh4AaABAg', 'reply_id': '', 'type': 'comment'}},
              {'text': 'Save the name of a hotel for the unmanned Kopal will say the name of a hotel', 'metadata': {'video_id': 'hlbzz7JTGrg', 'comment_id': 'UgxjmmGNufvBt8WZKqx4AaABAg', 'reply_id': '', 'type': 'comment'}},
              {'text': 'Thank you for all the important information. But since you have stayed in so many hotels in Darjeeling, can you suggest which hotel will be ideal for me? Can you guide which hotels have a balcony with a great view within a tariff of 30004000 per night during the offseason? I don39t need to be on the main road, but I am looking for a hotel with great balcony sight where we can sit and enjoy Kanchanjunga. I39m planning to visit Darjeeling in July 2023.', 'metadata': {'video_id': 'hlbzz7JTGrg', 'comment_id': 'UgxFCEILX3WFyHuoFFJ4AaABAg', 'reply_id': '', 'type': 'comment'}},
              {'text': 'Didi hotel price', 'metadata': {'video_id': 'eAZb8Rf8ZtM', 'comment_id': 'UgxzLYNfWAykUYWMZiV4AaABAg', 'reply_id': '', 'type': 'comment'}},
              {'text': 'Tariff? Car parking available?', 'metadata': {'video_id': 'eAZb8Rf8ZtM', 'comment_id': 'Ugy_tTLXfupQC9tut2t4AaABAg', 'reply_id': '', 'type': 'comment'}},
              {'text': 'Gajab hai diadi', 'metadata': {'video_id': 'eAZb8Rf8ZtM', 'comment_id': 'UgxuRYHjvXZsxVO8pV54AaABAg', 'reply_id': '', 'type': 'comment'}},
              {'text': 'Good to see promoting eating with hands Thankyou for visiting Darjeeling', 'metadata': {'video_id': 'eAZb8Rf8ZtM', 'comment_id': 'UgzTFpK031q_Qpx-MAp4AaABAg', 'reply_id': '', 'type': 'comment'}},
              {'text': 'Bro can you give the contact details??', 'metadata': {'video_id': 'v_ykMl8YeL4', 'comment_id': 'UgxatvaSviv9_gCxm354AaABAg', 'reply_id': '', 'type': 'comment'}},
              {'text': 'Is there AC inside the room?', 'metadata': {'video_id': 'v_ykMl8YeL4', 'comment_id': 'UgxQNga9xIi1OfHhXWl4AaABAg', 'reply_id': '', 'type': 'comment'}},
              {'text': 'Wat category of room?', 'metadata': {'video_id': 'v_ykMl8YeL4', 'comment_id': 'UgzabMKrioGdz6_uEJZ4AaABAg', 'reply_d': '', 'type': 'comment'}},
              {'text': 'What is the charge of this room', 'metadata': {'video_id': 'v_ykMl8YeL4', 'comment_id': 'UgwyxmJJblo_tb7w-Lx4AaABAg', 'reply_id': '', 'type': 'comment'}},
              {'text': 'Your voice yr', 'metadata': {'video_id': 'v_ykMl8YeL4', 'comment_id': 'Ugxcie2kJPMt5i3J1Zl4AaABAg', 'reply_id': '', 'type': 'comment'}},
              {'text': 'I like the room bro', 'metadata': {'video_id': 'v_ykMl8YeL4', 'comment_id': 'UgwHm7VphUIGjtuotKx4AaABAg', 'reply_id': '', 'type': 'comment'}},
              {'text': 'Hopefully will do a video for you all on Goa.', 'metadata': {'video_id': 'v_ykMl8YeL4', 'comment_id': 'UgyhvsryNfI0cKPWbU54AaABAg', 'reply_id': '', 'type': 'comment'}},
              {'text': 'Mam first book from hotel or counter.', 'metadata': {'video_id': 'sz5XgymzB88', 'comment_id': 'Ugxa4QFk9VSTgSAyUat4AaABAg', 'reply_id': '', 'type': 'comment'}},
              {'text': 'Mum bike parking?Will luggage be safe?What would be needed for marriage wife?', 'metadata': {'video_id': 'sz5XgymzB88', 'comment_id': 'Ugxmb0isw5msBwTxt9p4AaABAg', 'reply_id': '', 'type': 'comment'}},
              {'text': 'It will be booked online or offline ??', 'metadata': {'video_id': 'sz5XgymzB88', 'comment_id': 'UgyN1LIwYSbZjavDqQF4AaABAg', 'reply_id': '', 'type': 'comment'}},
              {'text': 'Unmarried couples allowed there?', 'metadata': {'video_id': 'sz5XgymzB88', 'comment_id': 'Ugx6FMZVRvu7P5axNaV4AaABAg', 'reply_id': '', 'type': 'comment'}}]

    # rag.ingest_vector(texts, collection_name="test_collection_7")
    ques = "Which restaurant is best?"
    ques = "What is wrong with the pronounciation?"
    ques = "What people are looking for?"
    ques = "What are the places the people are watching the video from?"
    #ans = rag.ingest_vector(text_array=texts1, collection_name="yt_collection_admin")
    ans = rag.answer_question(question=ques, context="", collection_name="yt_collection_admin")
    print(ans)