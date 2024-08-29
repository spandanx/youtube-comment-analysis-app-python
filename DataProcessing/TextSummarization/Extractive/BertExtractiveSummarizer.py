from summarizer import Summarizer

# Input text to be summarized
input_text = """
Your input text goes here. It can be a long paragraph or multiple paragraphs. 
"""

class BertExtractiveSummarizer:

    def __init__(self):
        # Create a BERT extractive summarizer
        self.summarizer = Summarizer()

    def summarizeText(self, text):
        summary = self.summarizer(text, min_length=50,
                             max_length=150)  # You can adjust the min_length and max_length parameters
        return summary

if __name__ == "__main__":
    text = """
    There are two ways of extracting text using TextRank: keyword and sentence extraction. 
    Keyword extraction can be done by simply using a frequency test, but this would almost always prove to be inaccurate. This is where TextRank automates the process to semantically provide far more accurate results based on the corpus.
    Sentence extraction, on the other hand, studies the corpus to summarize the most valid sentences pertaining to the subject matter and phonologically arranges it. 
    """
    bertExtractiveSummarizer = BertExtractiveSummarizer()
    print(bertExtractiveSummarizer.summarizeText(text))