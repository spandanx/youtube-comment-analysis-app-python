from transformers import pipeline

class QuestionAnswering:
    def __init__(self):
        self.question_answerer = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    def answer_question(self, question, context):
        return self.question_answerer(question=question, context=context)


if __name__ == "__main__":
  questionAnswering = QuestionAnswering()
  question = "Where do I live?"
  context = "My name is Merve and I live in İstanbul."
  answer = questionAnswering.answer_question(question=question, context=context)
  print(answer)