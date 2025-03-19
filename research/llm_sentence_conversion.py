

# from config.QDrant_setup import QDrantDB



if __name__ == '__main__':
    from langchain import PromptTemplate
    from langchain_ollama import OllamaLLM

    # qdrant_api_key = ""
    collection_name = "test_collection_4"


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

    from langchain.chains import LLMChain
    from langchain_core.prompts import PromptTemplate

    # template = "What is a good name for a company that makes {product}?"
    template = """Translate the below sentences to english and remove any special characters or emojis, 
                {sentences}
                """

    prompt = PromptTemplate.from_template(template)

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # generated = llm_chain.run(product="mechanical keyboard")
    sentences = """
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
    generated = llm_chain.run(sentences = sentences)
    print(generated)
