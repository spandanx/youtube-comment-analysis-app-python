import pandas as pd
import json
import pickle

char_file_path = "../data/lang_char/"

class OtherLanguageCharacterDetector:

    def __init__(self):
        self.char_maps = dict()
        self.langs = ["ben", "guj", "hin", "kan", "mal", "mar", "nep", "pan", "ori", "san", "tam", "tel", "urd"]
        for lang in self.langs:
            lang_output_path = char_file_path + lang + '_char.txt'
            self.load_char_maps(lang, lang_output_path)

    def load_char_maps(self, lang, char_file_path):
        with open(char_file_path, 'rb') as f:
            my_set = pickle.load(f)
        self.char_maps[lang] = my_set

    def detect_word_lang(self, word):
        if len(word) == 0:
            return {"letter_lang": "EMPTY", "meaning_lang": "EMPTY", "confidence": 100.0}
        word_set = set(word)
        for key, value in self.char_maps.items():
            if word[0] in value:
                union = word_set.union(word_set)
                parcentage = (len(union)/len(word))*100
                return {"letter_lang": key, "meaning_lang": key, "confidence": parcentage}
        return {"letter_lang": "NOT_FOUND", "meaning_lang": "NOT_FOUND", "confidence": 100.0}

    def detect_sentence_lang(self, sentence):
        words = sentence.split(' ')
        lst = []
        for word in words:
            current_word = {
                "word": word
            }
            current_word.update(self.detect_word_lang(word))
            lst.append(current_word)
        return lst


if __name__ == "__main__":
    lang_array = ["ben", "guj", "hin", "kan", "mal", "mar", "nep", "pan", "ori", "san", "tam", "tel", "urd"]
    otherLanguageCharacterDetector = OtherLanguageCharacterDetector()
    # for lang in lang_array:
    #     lang_output_path = char_file_path + lang + '_char.txt'
    #     otherLanguageCharacterDetector.load_char_maps(lang, lang_output_path)
    print(otherLanguageCharacterDetector.char_maps)
    word = "বসন্তের, ভ্রমণ, निर्माली"
    print(otherLanguageCharacterDetector.detect_sentence_lang(word))