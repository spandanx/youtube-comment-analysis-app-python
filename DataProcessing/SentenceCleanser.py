import re

class SentenceCleanser:
    def process_sentence(self, text):
        text = text.replace("<br>", " ")
        text = re.sub(' +', ' ', text)
        return text

    def remove_special_chars(self, text):
        text = re.sub('[^A-Za-z0-9,.?! ]+', '', text)
        text = re.sub(' +,*', ' ', text)
        text = re.sub(' +', ' ', text)
        return text

if __name__ == "__main__":
    sentenceCleanser = SentenceCleanser()
    sentence = "While consulting, ; : I sometimes tell people about the consulting business. ? ! 🤣🤣🤣, 👍,  😍 ... ...."
    print(sentenceCleanser.remove_special_chars(sentence))