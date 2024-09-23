import pandas as pd
import json
import pickle

# data_path = "C:/Users/spand/Downloads/Compressed/Transliteration Dataset 21 Indic languages/data/ben/ben_test.json"
data_path = "C:/Users/spand/Downloads/Compressed/Transliteration Dataset 21 Indic languages/data/"
output_path = "../../data/lang_char/"
sample_size = 10000
class OtherLanguageCharacterGenerator:

    def generate_character_set(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        char_set = set()
        for line in lines[: min(sample_size, len(lines) - 1)]:
            each_line = json.loads(line)
            # print(each_line)
            native_word = each_line["native word"]
            for char_data in native_word:
                char_set.add(char_data)
        return char_set

    def write_to_file(self, char_set, output_path):
        with open(output_path, 'wb') as f:
            pickle.dump(char_set, f)

        ##To read it again from file
        # with open('kos.txt', 'rb') as f:
        #     my_set = pickle.load(f)


if __name__ == "__main__":
    lang_array = ["ben", "guj", "hin", "kan", "mal", "mar", "nep", "pan", "ori", "san", "tam", "tel", "urd"]
    for lang in lang_array:
        lang_data_path = data_path + lang + '/' + lang + '_train.json'
        lang_output_path = output_path + lang + '_char.txt'
        otherLanguageCharacterGenerator = OtherLanguageCharacterGenerator()
        char_set = otherLanguageCharacterGenerator.generate_character_set(lang_data_path)
        otherLanguageCharacterGenerator.write_to_file(char_set, lang_output_path)
        print(len(char_set))
        print(char_set)
        # print(type(char_set))
