from sumy.summarizers.lsa import LsaSummarizer
summarizer_lsa = LsaSummarizer()

#Plain text parsers since we are parsing through text
from sumy.parsers.plaintext import PlaintextParser

#for tokenization
from sumy.nlp.tokenizers import Tokenizer

# file = "001.txt"
st = """
There are two ways of extracting text using TextRank: keyword and sentence extraction. 
Keyword extraction can be done by simply using a frequency test, but this would almost always prove to be inaccurate. This is where TextRank automates the process to semantically provide far more accurate results based on the corpus.
Sentence extraction, on the other hand, studies the corpus to summarize the most valid sentences pertaining to the subject matter and phonologically arranges it. 
"""
# parser = PlaintextParser.from_file(file, Tokenizer("english"))
parser = PlaintextParser.from_string(st, Tokenizer("english"))


# Summarize using sumy LSA
summary =summarizer_lsa(parser.document,2)
lsa_summary=""
for sentence in summary:
    lsa_summary+=str(sentence)

print("SUMMARY - ")
print(lsa_summary)