from transformers import pipeline

class DistilbertTextSummarizer:

  def __init__(self):
    self.summarizer = pipeline('summarization')

  def summarizeText(self, text):
    # summarizer = pipeline('summarization')
    summary = self.summarizer(text)
    return summary


if __name__ == "__main__":
  distilbertTextSummarizer = DistilbertTextSummarizer()
  text = """
  There are two ways of extracting text using TextRank: keyword and sentence extraction. 
  Keyword extraction can be done by simply using a frequency test, but this would almost always prove to be inaccurate. This is where TextRank automates the process to semantically provide far more accurate results based on the corpus.
  Sentence extraction, on the other hand, studies the corpus to summarize the most valid sentences pertaining to the subject matter and phonologically arranges it. 
  """
  # question = "Where do I live?"
  # context = "My name is Merve and I live in İstanbul."
  # answer = textSummarizer.answer_question(question=question, context=context)
  print(distilbertTextSummarizer.summarizeText(text))
  # print(answer)

