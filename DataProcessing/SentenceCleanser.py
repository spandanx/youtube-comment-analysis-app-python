import re

class SentenceCleanser:
    def process_sentence(self, text):
        text = text.replace("<br>", " ")
        text = re.sub(' +', ' ', text)
        return text

