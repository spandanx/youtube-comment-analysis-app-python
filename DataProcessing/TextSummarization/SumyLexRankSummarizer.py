from sumy.summarizers.lex_rank import LexRankSummarizer
summarizer_lex = LexRankSummarizer()

#Plain text parsers since we are parsing through text
from sumy.parsers.plaintext import PlaintextParser

#for tokenization
from sumy.nlp.tokenizers import Tokenizer

# file = "001.txt"

class SumyLexRankSummarizer:

    def summarizeText(self, text):
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        # parser = PlaintextParser.from_file(file, Tokenizer("english"))

        # Summarize using sumy LexRank
        summary= summarizer_lex(parser.document, 2)
        lex_summary=""
        for sentence in summary:
            lex_summary+=str(sentence)
        return lex_summary

if __name__ == "__main__":
    print("NLTK SUMMARY - ")
    text = """
    There are two ways of extracting text using TextRank: keyword and sentence extraction. 
    Keyword extraction can be done by simply using a frequency test, but this would almost always prove to be inaccurate. This is where TextRank automates the process to semantically provide far more accurate results based on the corpus.
    Sentence extraction, on the other hand, studies the corpus to summarize the most valid sentences pertaining to the subject matter and phonologically arranges it. 
    """
    sumyLexRankSummarizer = SumyLexRankSummarizer()
    print(sumyLexRankSummarizer.summarizeText(text))


# print(lex_summary)