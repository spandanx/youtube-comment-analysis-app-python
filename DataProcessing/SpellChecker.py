from spellchecker import SpellChecker
spell = SpellChecker()  # loads default word frequency list
#
# words = list(spell.word_frequency.words())
# print(len(list(spell.word_frequency.words())))
# spell.word_frequency.remove_words(words)
# print(len(list(spell.word_frequency.words())))
#
# spell.word_frequency.load_words(['microsoft', 'apple', 'google'])
# res = spell.unknown(['microsoft'])  # will return both now!
# print(res)

res = spell.correction('applee')
print(res)

import pandas as pd
# import json
# import pickle
#
# # data_path = "C:/Users/spand/Downloads/Compressed/Transliteration Dataset 21 Indic languages/data/ben/ben_test.json"
# data_path = "C:/Users/spand/Downloads/Compressed/Transliteration Dataset 21 Indic languages/data/"
# output_path = "../data/spell_checker/"
# sample_size = 10000
# class CustomSpellChecker:
#
#     def __init__(self):
#         self.spell = SpellChecker()
#         words = list(self.spell.word_frequency.words())
#         self.spell.word_frequency.remove_words(words)
#
#     def generate_character_set(self, data_path):
#         with open(data_path, 'r', encoding='utf-8') as f:
#             lines = f.read().split('\n')
#         char_set = set()
#         for line in lines[: min(sample_size, len(lines) - 1)]:
#             each_line = json.loads(line)
#             # print(each_line)
#             native_word = each_line["english word"]
#             # for char_data in native_word:
#             char_set.add(native_word)
#         return char_set
#
#     def write_to_file(self, output_path, spell_obj):
#         with open(output_path, 'wb') as f:
#             pickle.dump(spell_obj, f)
#
#     def create_custom_spell_dict(self, data_path, output_path):
#         char_set = self.generate_character_set(data_path)
#         char_list = list(char_set)
#         self.spell.word_frequency.load_words(char_list)
#         self.write_to_file(output_path, self.spell)
#
# if __name__ == "__main__":
#     lang_array = ["ben"]
#     # lang_array = ["ben", "guj", "hin", "kan", "mal", "mar", "nep", "pan", "ori", "san", "tam", "tel", "urd"]
#     for lang in lang_array:
#         lang_data_path = data_path + lang + '/' + lang + '_train.json'
#         lang_output_path = output_path + lang + '_char.txt'
#         customSpellChecker = CustomSpellChecker()
#         char_set = customSpellChecker.create_custom_spell_dict(lang_data_path, lang_output_path)
#         # print(type(char_set))
