from spellchecker import SpellChecker
# spell = SpellChecker()  # loads default word frequency list
#
# words = list(spell.word_frequency.words())
# print(len(list(spell.word_frequency.words())))
# spell.word_frequency.remove_words(words)
# print(len(list(spell.word_frequency.words())))
#
# spell.word_frequency.load_words(['microsoft', 'apple', 'google'])
# res = spell.unknown(['microsoft', 'google', 'man', 'appple', 'eat'])  # will return both now!
# print(res)

import pandas as pd
import json
import pickle

# data_path = "C:/Users/spand/Downloads/Compressed/Transliteration Dataset 21 Indic languages/data/ben/ben_test.json"
data_path = "C:/Users/spand/Downloads/Compressed/Transliteration Dataset 21 Indic languages/data/"
output_path = "../data/spell_checker/"
sample_size = 10000
class CustomSpellCheckerDetector:

    def set_lang(self, lang):
        self.lang_array = lang

    def load_spell_checker(self, spell_checker_path):
        with open(spell_checker_path, 'rb') as fp:
            return pickle.load(fp)

    def load_all_spell_chckers_obj(self, spell_checker_base_path):
        self.spell_obj_map = {}
        for lang in self.lang_array:
            spell_checker_lang_path = spell_checker_base_path + lang + '_char.txt'
            spell_obj = self.load_spell_checker(spell_checker_lang_path)
            self.spell_obj_map[lang] = spell_obj

    def detect_word(self, word):
        possible_lang = []
        for lang in self.lang_array:
            unknown_word = list(self.spell_obj_map[lang].unknown([word]))
            if len(unknown_word)>0:
                corrected = self.spell_obj_map[lang].correction(unknown_word[0])
                if corrected is not None:
                    possible_lang.append({"corrected": corrected, "lang": lang, "original": word, "known": False})
                    return possible_lang
            else:
                possible_lang.append({"corrected": word, "lang": lang, "original": word, "known": True})
                return possible_lang
        return [{"corrected": word, "lang": "NOT_FOUND", "original": word, "known": False}]


if __name__ == "__main__":
    lang_array = ["ben"]
    customSpellCheckerDetector = CustomSpellCheckerDetector()
    customSpellCheckerDetector.set_lang(lang_array)
    customSpellCheckerDetector.load_all_spell_chckers_obj(output_path)
    word = "panthe"
    print(customSpellCheckerDetector.detect_word(word))
    # lang_array = ["ben", "guj", "hin", "kan", "mal", "mar", "nep", "pan", "ori", "san", "tam", "tel", "urd"]
