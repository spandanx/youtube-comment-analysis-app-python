import requests

# Input text to be summarized
input_text = """
Your input text goes here. It can be a long paragraph or multiple paragraphs. 
"""
class TextTeaserSummarizer:

    def summarizeText(self, text):
        response = requests.post("http://www.textteaser.com/api", data={"text": text})
        summary = response.text
        return summary

if __name__ == "__main__":
    text = """
    There are two ways of extracting text using TextRank: keyword and sentence extraction. 
    Keyword extraction can be done by simply using a frequency test, but this would almost always prove to be inaccurate. This is where TextRank automates the process to semantically provide far more accurate results based on the corpus.
    Sentence extraction, on the other hand, studies the corpus to summarize the most valid sentences pertaining to the subject matter and phonologically arranges it. 
    """
    textTeaserSummarizer = TextTeaserSummarizer()
    print(textTeaserSummarizer.summarizeText(text))