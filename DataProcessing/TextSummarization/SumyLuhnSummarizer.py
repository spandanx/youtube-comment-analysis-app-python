from sumy.summarizers.luhn import LuhnSummarizer

#Plain text parsers since we are parsing through text
from sumy.parsers.plaintext import PlaintextParser

#for tokenization
from sumy.nlp.tokenizers import Tokenizer


class SumyLuhnSummarizer:

    def summarizeText(self, text):
        parser = PlaintextParser.from_string(text, Tokenizer("english"))


        summarizer_1 = LuhnSummarizer()
        summary_1 =summarizer_1(parser.document,2)

        text_summary = ""
        for sentence in summary_1:
            # print(sentence)
            text_summary += str(sentence)
        return text_summary

if __name__ == "__main__":
    print("NLTK SUMMARY - ")
    text = """
    There are two ways of extracting text using TextRank: keyword and sentence extraction. 
    Keyword extraction can be done by simply using a frequency test, but this would almost always prove to be inaccurate. This is where TextRank automates the process to semantically provide far more accurate results based on the corpus.
    Sentence extraction, on the other hand, studies the corpus to summarize the most valid sentences pertaining to the subject matter and phonologically arranges it. 
    """
    sumyLuhnSummarizer = SumyLuhnSummarizer()
    print(sumyLuhnSummarizer.summarizeText(text))