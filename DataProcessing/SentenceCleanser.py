import re

class SentenceCleanser:
    def process_sentence(self, text):
        text = self.clean_square_brackets(text)
        text = text.replace("<br>", " ")
        text = re.sub('<a.*</a>', ' ', text)
        text = text.replace("</a>", " ")
        text = re.sub('https?://\S+', '', text) # remove links
        text = re.sub('@@\S+', '', text) #remove mentions
        text = re.sub(' +', ' ', text)
        text = " ".join(text.split())
        return text

    def remove_special_chars(self, text):
        text = re.sub('[^A-Za-z0-9,.?! ]+', '', text)
        text = re.sub(' +,*', ' ', text)
        text = re.sub(' +', ' ', text)
        text = " ".join(text.split())
        return text

    '''
    Removes the square brackets from the text.
    '''
    def clean_square_brackets(self, text):
        cleaned_text = ""
        inside_bracket = False
        for character in text:
            if character == "[":
                inside_bracket = True
            if not inside_bracket:
                cleaned_text += character
            if character == "]":
                inside_bracket = False
        return cleaned_text

if __name__ == "__main__":
    sentenceCleanser = SentenceCleanser()
    # sentence = "While consulting, ; : I sometimes tell people about the consulting business. ? ! 🤣🤣🤣, 👍,  😍 ... ...."
    # print(sentenceCleanser.remove_special_chars(sentence))
    # sentence = '</a><br>Facebook <a href="https://www.facebook.com/Debdutyoutube/">https://www.facebook.com/Debdutyoutube/</a>'
    sentence = 'http://youtu.be/CMnOg1hfX4M Durga Puja video'
    # sentence = '@@ganeshbasak6272 Watch my Top 5 Best Durga Puja'
    print(sentenceCleanser.process_sentence(sentence))