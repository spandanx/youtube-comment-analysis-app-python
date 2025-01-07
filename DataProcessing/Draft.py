# from DataProcessing.TextSummarization.Abstractive.BARTAbstractiveSummarizer import BARTAbstractiveSummarizer
# from DataProcessing.TextSummarization.Abstractive.DistilbertSummarizer import DistilbertSummarizer
# from DataProcessing.TextSummarization.Abstractive.T5BaseSummarizer import T5BaseSummarizer
from DataProcessing.TextSummarization.Abstractive.T5SmallSummarizer import T5SmallSummarizer
# from DataProcessing.TextSummarization.Extractive.BertExtractiveSummarizer import BertExtractiveSummarizer
# from DataProcessing.TextSummarization.Extractive.NLTKSummarizer import NLTKSummarizer
# from DataProcessing.TextSummarization.Extractive.SumyLSASummarizer import SumyLSARankSummarizer
# from DataProcessing.TextSummarization.Extractive.SumyLexRankSummarizer import SumyLexRankSummarizer
# from DataProcessing.TextSummarization.Extractive.SumyLuhnSummarizer import SumyLuhnSummarizer
# from DataProcessing.TextSummarization.Extractive.SumyTextRankSummarizer import SumyTextRankSummarizer
from time import time

if __name__ == "__main__":
    sentences = [
            'You probably havent seen Chor Bagan...near. mg metro.. one of finest pandal I bet. I say, grandpa, your drone is fine now. Dada ami I am saying Shubojit Paul contact a basket ki lures',
            'Happy durga puja sir. Happy Durga Pujo Shubh Panchami. Dada, wha north Kolkata, another and big Puja visit Will do it, Nav para dada Y Sangha Baranagar. Ehaka thim hei Introduction Look, I guarantee it. ki you like it. Durga Puja video. go mom Durga. Kharagpur Durga Puja Pandal 2024. go mom Durga Jai maa Durga ',
            'Kalyani, West Bengal, Nadia district',
            'Coming to kolkata on 5th oct, kindly guide us which pandal to visit.. as its last day. Chorbagan ta top 10 a It would be better to keep it. Kolkata is most important city in India. Patna ka. Dhono dhonne puspe vora. It,s poem on rabindra nath thakur. Jay maa durga. Jay eye di ',
            'Hope You Enjoyed The Video Add Me on Social Media Instagram. Thanks for watching Add Me on Social Media Instagram'
        ]
    # nltkSummarizer = NLTKSummarizer()
    # sumyLexRankSummarizer = SumyLexRankSummarizer()
    # sumyLSARankSummarizer = SumyLSARankSummarizer()
    # sumyLuhnSummarizer = SumyLuhnSummarizer()
    # sumyTextRankSummarizer = SumyTextRankSummarizer()
    # ts = TextSummarizer()
    # distilbertSummarizer = DistilbertSummarizer()
    # bertExtractiveSummarizer = BertExtractiveSummarizer()
    # bartAbstractiveSummarizer = BARTAbstractiveSummarizer()
    # t5BaseSummarizer = T5BaseSummarizer()
    t5SmallSummarizer = T5SmallSummarizer()

    # print(distilbertSummarizer.summarizeText(sentences[0]))
    models = [
                # ["ts", ts],
                # ["bertExtractiveSummarizer", bertExtractiveSummarizer],
                # ["nltkSummarizer", nltkSummarizer],
                # ["sumyLexRankSummarizer", sumyLexRankSummarizer],
                # ["sumyLSARankSummarizer", sumyLSARankSummarizer],
                # ["sumyLuhnSummarizer", sumyLuhnSummarizer],
                # ["sumyTextRankSummarizer", sumyTextRankSummarizer],

                # ["bartAbstractiveSummarizer", bartAbstractiveSummarizer],
                # ["distilbertSummarizer", distilbertSummarizer],
                # ["t5BaseSummarizer", t5BaseSummarizer],
                ["t5SmallSummarizer", t5SmallSummarizer]
             ]
    summary_map = {}
    for model_name, model in models:
        t = time()
        summary_map[model_name] = []
        for sentence in sentences:
            # print("############")
            # print(sentence)
            summary = model.summarizeText(sentence)
            summary_map[model_name].append(summary)
            # print(summary)
        t1 = time()
        print(model_name, t1-t)

    ##################
    for i in range(len(sentences)):
        print("TEXT - ")
        print(sentences[i])
        print("SUMMARY - ")
        for model_name, model in models:
            print("\t", model_name + " - ", summary_map[model_name][i])

    # old_comments = {'statements': ['Happy durga puja sir', 'Happy Durga Happy Puja Panchami', 'You probably havent seen Chor Bagan...near. mg metro.. one of finest pandal I bet', 'Dada, wha north Kolkata, another and big Puja visit Will do it, Nav para dada Y Sangha Baranagar. Ehaka thim hei Introduction Look, I guarantee it. ki you like it', 'Coming to kolkata on 5th oct, kindly guide us which pandal to visit.. as its last day', 'Chorbagan ta top 10 a It would be better to keep it', 'Durga Puja video ', '', 'Kalyani, West Bengal, Nadia district', 'go mom Durga ', 'Jai Maa Durga . Har Har Mahadev ', 'Bhai background music from words download koro', 'Hope You Enjoyed The Video Add Me on Social Media Instagram', 'Dhono dhonne puspe vora. It,s poem on rabindra nath thakur.', 'Thanks for watching Add Me on Social Media Instagram', 'Jay maa durga ', 'Dada ami I am saying Shubojit Paul contact a basket ki lures', 'go mom Durga Jai maa Durga ', 'Jay eye di '], 'questions': ['Wishing You Happy Durga Puja', 'Watch my Top 5 Best Durga Puja', 'helo please make the beautiful procession of maa durga immersion on the streets of kolkata this year on 12 13 and 14 october both north and south kolkata In North kolkata it will mainly take place near hedua park or beadon street 15 20 minutes from hatibagan star theatre but I dont know the way of south kolkata procession please find or search the place where it will take place exactly and do the vlog thank u ...', 'helo please make the beautiful procession of maa durga immersion on the streets of kolkata this year on 12 13 and 14 october both north and south kolkata .. thank u ...']}
    # new_comments = {'statements': ['Happy durga puja sir', 'Happy Durga Puja Happy Panchami', 'You may not have seen Chor Bagan...Near Metro...One of the Finest Pandal Bet', 'Dada, you from North Kolkata, will visit another big Puja, now for Dada or Sangha Baranagar. This is the Introduction that I guarantee you will be impressed by the time you watch it.', 'Coming to kolkata on 5th oct, kindly guide us which pandal to visit.. as its last day', 'Those who keep Chorbagans top 10', 'Durga Puja video ', '', 'Kalyani, West Bengal, Nadia district', 'Jai maa Durga ', 'Jai Maa Durga . Har Har Mahadev ', 'Bhai background music download from Katha', 'Hope You Enjoyed The Video Add Me on Social Media Instagram', 'Dhone dhonne puspe vora. Its poem on rabindranath thakur.', 'Thanks for watching Add Me on Social Media Instagram', 'Jay maa durga ', 'Dada I am Shubhjit Paul saying how to contact', 'Jai maa Durga Jai maa Durga ', 'Jay eyes on '], 'questions': ['Wishing You Happy Durga Puja', 'Watch my Top 5 Best Durga Puja', 'helo please make the beautiful procession of maa durga immersion on the streets of kolkata this year on 12 13 and 14 october both north and south kolkata In North kolkata it will mainly take place near hedua park or beadon street 15 20 minutes from hatibagan star theatre but I dont know the way of south kolkata procession please find or search the place where it will take place exactly and do the vlog thank u ...', 'helo please make the beautiful procession of maa durga immersion on the streets of kolkata this year on 12 13 and 14 october both north and south kolkata .. thank u ...']}
    #
    # for old_, new_ in zip(old_comments["statements"], new_comments["statements"]):
    #     print("STATEMENTS ---------------")
    #     print("OLD: ", old_)
    #     print("NEW: ", new_)
    #
    # for old_, new_ in zip(old_comments["questions"], new_comments["questions"]):
    #     print("QUESTIONS ---------------")
    #     print("OLD: ", old_)
    #     print("NEW: ", new_)