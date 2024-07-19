import pandas as pd
import json
import pickle

from ModelGenerator.TransliterationIndicLatin2Native import Transliteration_en_latin_bn
from TransliterationLanguageGeneraterDetecter.TransliterationLanguageDetector import TransliterationLanguageDetector

char_file_path = "../data/lang_char/"

class OtherLanguageCharacterDetector:

    def __init__(self):
        self.char_maps = dict()
        self.langs = ["ben", "guj", "hin", "kan", "mal", "mar", "nep", "pan", "ori", "san", "tam", "tel", "urd", "eng"]
        for lang in self.langs:
            lang_output_path = char_file_path + lang + '_char.txt'
            self.load_char_maps(lang, lang_output_path)

    def set_transliteration_object(self, obj):
        self.e2btrans_object = obj

    def load_char_maps(self, lang, char_file_path):
        with open(char_file_path, 'rb') as f:
            my_set = pickle.load(f)
        self.char_maps[lang] = my_set

    def detect_word_lang(self, word):
        if len(word) == 0:
            return {"letter_lang": "EMPTY", "confidence": 100.0}
        word_set = set(word)
        for key, value in self.char_maps.items():
            if word[0] in value:
                intersection = word_set.intersection(value)
                percentage = (len(intersection)/len(word_set))*100
                if key == "eng":
                    transliterationLanguageDetector = TransliterationLanguageDetector()
                    res = transliterationLanguageDetector.detect_word_lang(word)
                    meaning_lang = res["meaning_lang"]
                    if meaning_lang == "ben":
                        transed_word = self.e2btrans_object.transliterate_to_native(word)
                        print(transed_word)
                        res.update({"transliterated_word": transed_word})
                        return res
                return {"letter_lang": key, "confidence": percentage}
        return {"letter_lang": "NOT_FOUND", "confidence": 100.0}

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
    otherLanguageCharacterDetector = OtherLanguageCharacterDetector()
    tl = Transliteration_en_latin_bn()
    otherLanguageCharacterDetector.set_transliteration_object(tl)
    # for lang in lang_array:
    #     lang_output_path = char_file_path + lang + '_char.txt'
    #     otherLanguageCharacterDetector.load_char_maps(lang, lang_output_path)
    print(otherLanguageCharacterDetector.char_maps)
    word = "siddhanto বসন্তের ভ্রমণ निर्माली there"
    print(otherLanguageCharacterDetector.detect_sentence_lang(word))