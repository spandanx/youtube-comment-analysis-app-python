from transformers import pipeline

question_answerer = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

class TextSummarizer:

  def __init__(self):
    self.question_answerer = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    self.summarizer = pipeline('summarization')

  def summarizeText(self, text):
    # summarizer = pipeline('summarization')
    summary = self.summarizer(text)
    return summary

  def answer_question(self, question, context):
    return self.question_answerer(question=question, context=context)

if __name__ == "__main__":
  textSummarizer = TextSummarizer()
  # text = "It is known from archaeological evidence that a highly sophisticated urbanized culture—the Indus civilization—dominated the northwestern part of the subcontinent from about 2600 to 2000 bce. From that period on, India functioned as a virtually self-contained political and cultural arena, which gave rise to a distinctive tradition that was associated primarily with Hinduism, the roots of which possibly can be traced to the Indus civilization. Other religions, notably Buddhism and Jainism, originated in India—though their presence there is now quite small—and throughout the centuries residents of the subcontinent developed a rich intellectual life in such fields as mathematics, astronomy, architecture, literature, music, and the fine arts."
  # summarizedText = textSummarizer.summarizeText(text)
  # print(summarizedText)
  question = "Where do I live?"
  context = "My name is Merve and I live in İstanbul."
  answer = textSummarizer.answer_question(question=question, context=context)
  print(answer)

