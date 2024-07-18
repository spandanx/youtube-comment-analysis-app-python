from time import time
import enchant

from CharacterLanguageGeneraterDetecter.OtherLanguageCharacterDetector import OtherLanguageCharacterDetector

d = enchant.Dict("en_US")
# import nltk
# nltk.download()
from nltk.corpus import words
nltk_words = set(words.words())
text_path = "C:\\Users\\spand\\OneDrive\\Documents\\Sample_english_text.txt"

class EnglishLanguageDetector:
    def detect_language_of_text(self, lines):
        # nltk_wordset = set()
        # enchant_wordset = set()
        lang_list = []
        for line in lines:
            print(line)
            word_list = []
            words = line.split(' ')
            for word in words:
                if len(word)==0:
                    continue
                current_word = {
                    "word": word
                }
                if d.check(word):
                    print("Present - ", word)
                    # enchant_wordset.add(string)
                    current_word.update({"letter_lang": "english", "meaning_lang": "english", "confidence": 100.0})
                else:
                    current_word.update(otherLanguageCharacterDetector.detect_word_lang(word))

                word_list.append(current_word)
            lang_list.append(word_list)
        return lang_list

        # if string in nltk_words:
        #     nltk_wordset.add(string)
if __name__ == "__main__":
    otherLanguageCharacterDetector = OtherLanguageCharacterDetector()
    file = open(text_path, "r")
    content = file.read()
    file.close()

    lines = content.split('\n')

    t = time()

    lines = [
        "are more simply constructed from the 7-orthoplex.",
        "are farmers, while an additional 5% receives their livelihood from raising livestock.",
        "বসন্তের, ভ্রমণ, निर्माली",
        "nirmaali shivaalapurva siddhanto"
    ]

    englishLanguageDetector = EnglishLanguageDetector()
    lang_list = englishLanguageDetector.detect_language_of_text(lines)
    t1 = time()
    print(t1 - t)
    for line in lang_list:
        print(line)
# for wd in enchant_wordset:
#     if wd not in nltk_wordset:
#         print(wd)