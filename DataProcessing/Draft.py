from DataProcessing.TextSummarization.Abstractive.BARTAbstractiveSummarizer import BARTAbstractiveSummarizer
from DataProcessing.TextSummarization.Abstractive.DistilbertTextSummarizer import DistilbertTextSummarizer
from DataProcessing.TextSummarization.Abstractive.T5BaseSummarizer import T5BaseSummarizer
from DataProcessing.TextSummarization.Abstractive.T5SmallSummarizer import T5SmallSummarizer
from DataProcessing.TextSummarization.Extractive.BertExtractiveSummarizer import BertExtractiveSummarizer
from DataProcessing.TextSummarization.Extractive.NLTKSummarizer import NLTKSummarizer
from DataProcessing.TextSummarization.Extractive.SumyLSASummarizer import SumyLSARankSummarizer
from DataProcessing.TextSummarization.Extractive.SumyLexRankSummarizer import SumyLexRankSummarizer
from DataProcessing.TextSummarization.Extractive.SumyLuhnSummarizer import SumyLuhnSummarizer
from DataProcessing.TextSummarization.Extractive.SumyTextRankSummarizer import SumyTextRankSummarizer
from time import time

if __name__ == "__main__":
    sentences = [
            'You probably havent seen Chor Bagan...near. mg metro.. one of finest pandal I bet. I say, grandpa, your drone is fine now. Dada ami I am saying Shubojit Paul contact a basket ki lures',
            'Happy durga puja sir. Happy Durga Pujo Shubh Panchami. Dada, wha north Kolkata, another and big Puja visit Will do it, Nav para dada Y Sangha Baranagar. Ehaka thim hei Introduction Look, I guarantee it. ki you like it. Durga Puja video. go mom Durga. Kharagpur Durga Puja Pandal 2024. go mom Durga Jai maa Durga ',
            'Kalyani, West Bengal, Nadia district',
            'Coming to kolkata on 5th oct, kindly guide us which pandal to visit.. as its last day. Chorbagan ta top 10 a It would be better to keep it. Kolkata is most important city in India. Patna ka. Dhono dhonne puspe vora. It,s poem on rabindra nath thakur. Jay maa durga. Jay eye di ',
            'Hope You Enjoyed The Video Add Me on Social Media Instagram. Thanks for watching Add Me on Social Media Instagram'
        ]
    nltkSummarizer = NLTKSummarizer()
    sumyLexRankSummarizer = SumyLexRankSummarizer()
    sumyLSARankSummarizer = SumyLSARankSummarizer()
    sumyLuhnSummarizer = SumyLuhnSummarizer()
    sumyTextRankSummarizer = SumyTextRankSummarizer()
    # ts = TextSummarizer()
    distilbertTextSummarizer = DistilbertTextSummarizer()
    bertExtractiveSummarizer = BertExtractiveSummarizer()
    bartAbstractiveSummarizer = BARTAbstractiveSummarizer()
    t5BaseSummarizer = T5BaseSummarizer()
    t5SmallSummarizer = T5SmallSummarizer()
    models = [
                # ["ts", ts],
                ["bertExtractiveSummarizer", bertExtractiveSummarizer],
                ["nltkSummarizer", nltkSummarizer],
                ["sumyLexRankSummarizer", sumyLexRankSummarizer],
                ["sumyLSARankSummarizer", sumyLSARankSummarizer],
                ["sumyLuhnSummarizer", sumyLuhnSummarizer],
                ["sumyTextRankSummarizer", sumyTextRankSummarizer],

                ["bartAbstractiveSummarizer", bartAbstractiveSummarizer],
                ["distilbertTextSummarizer", distilbertTextSummarizer],
                ["t5BaseSummarizer", t5BaseSummarizer],
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
