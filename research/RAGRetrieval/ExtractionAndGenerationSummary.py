# from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings  # to get embeddings
from langchain.vectorstores import Qdrant  # vector database
from qdrant_client import QdrantClient
from langchain.llms import CTransformers  # to get llm
from langchain.text_splitter import RecursiveCharacterTextSplitter  # splitting text into chunks
from langchain.chains import RetrievalQA  # building RAGRetrieval chain
from langchain.document_loaders import PyPDFLoader, UnstructuredURLLoader  # to read pdfs, urls
# from langchain_community.llms import OpenLLM
from langchain_ollama import OllamaLLM

qdrant_url = "http://180.188.226.161:6333"

collection_name = "test_collection_5"

def extract_and_generate(context):
    # a custom prompt help us to assist our agent with better answer and make sure to not make up answers
    custom_prompt_template = """Use the following pieces of information to answer the user’s question.
    If you don’t know the answer, just say that you don’t know, don’t try to make up an answer.
    
    Context: {context}
    Question: {question}
    
    Only return the helpful answer below and nothing else.
    Helpful and Caring answer:
    """

    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])


    llm = OllamaLLM(model="llama3.2:1b")


    # template = "What is a good name for a company that makes {product}?"
    template = """Summarize the context below 
                Context: {context}
                """

    prompt = PromptTemplate.from_template(template)

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # generated = llm_chain.run(product="mechanical keyboard")

    generated = llm_chain.run(context = context)
    return generated

if __name__ == '__main__':
    context = """
            'Thriller Ba Horror, Audio Story Gula Sunte Onk Valo Lage Amar.Ovinoy r thriller station er audio er moto eto crystal clear sound er sathe background music er combination, and jara voice den tader moto sundor vabe upostapon kono chano chano chano radio shadio shady.kothaw pai nai.Na .. Ekta kmn niramis niramish typer lage amr kache personally .. Stoty jotoi romamcokor hok na kno sunte gele r Valo Lge na.Sunechi.bepar .. tar poreo request yhakbe apnara jodi r ekyu taratari story dite paren ... thanks thriller station ..',
            'Excellent story &amp; presentation',
            'Ashonkhyo dhonyobaad o bhalobasha apnake! Credit goes to the sound designer babu sarkar for the zd audio effCTs & amp;Quality!amader channel ti ke egiye niye jte sahajyo korben! 🙂🙂🤟🤟',
            'Thanx a lot!Plz do like,share,subscribe n help our channel to grow!🙂🙂🤟🤟', 'Think this is nice, think awesome',
            'Radio Marchi for Moto Valona', 'Excellent story &amp; presentation',
            'Ashonkhyo dhonyobaad o bhalobasha apnake! Credit goes to the sound designer babu sarkar for the zd audio effCTs & amp;Quality!amader channel ti ke egiye niye jte sahajyo korben! 🙂🙂🤟🤟',
            'Thanx a lot!Plz do like,share,subscribe n help our channel to grow!🙂🙂🤟🤟', 'Think this is nice, think awesome',
            'Radio Marchi for Moto Valona',
            'Each one of you performs beautiful and thank everyone.Especially the voice and acting of Raja Bhattacharya is imperative.Want to hear more stories.',
            'Malda 🙂', 'Darun', 'Mr. Haldar moshai is so generous and have big brave heart...salute him',
            'Sound ears in the sound, oh boring', 'Colonial voice very good'
        """
    res = extract_and_generate(context)
    print(res)