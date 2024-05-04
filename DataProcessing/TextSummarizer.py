from transformers import pipeline

class TextSummarizer:
  def summarizeText(self, text):
    summarizer = pipeline('summarization')
    summary = summarizer(text)
    return summary
