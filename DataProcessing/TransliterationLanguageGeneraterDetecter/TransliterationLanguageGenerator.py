import pandas as pd
import json
import pickle

data_path = "C:/Users/spand/Downloads/Compressed/Transliteration Dataset 21 Indic languages/data/"
output_path = "../../data/transliteration_words/"
sample_size = 10000
class TransliterationLanguageGenerator:

    def generate_word_set(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        word_set = set()
        for line in lines[: min(sample_size, len(lines) - 1)]:
            each_line = json.loads(line)
            # print(each_line)
            eng_word = each_line["english word"]
            word_set.add(eng_word)
        return word_set

    def write_to_file(self, word_set, output_path):
        with open(output_path, 'wb') as f:
            pickle.dump(word_set, f)

if __name__ == "__main__":
    transliterationLanguageGenerator = TransliterationLanguageGenerator()
    lang_array = ["ben", "guj", "hin", "kan", "mal", "mar", "nep", "pan", "ori", "san", "tam", "tel", "urd"]
    # lang_array = ["hin"]
    for lang in lang_array:
        lang_data_path = data_path + lang + '/' + lang + '_train.json'
        lang_output_path = output_path + 'transliteration_words_' + lang + '.txt'
        word_set = transliterationLanguageGenerator.generate_word_set(lang_data_path)
        print(len(word_set))
        lang_data_path = data_path + lang + '/' + lang + '_test.json'
        word_set2 = transliterationLanguageGenerator.generate_word_set(lang_data_path)
        word_set = word_set.union(word_set2)
        print(len(word_set))
        lang_data_path = data_path + lang + '/' + lang + '_valid.json'
        word_set3 = transliterationLanguageGenerator.generate_word_set(lang_data_path)
        word_set = word_set.union(word_set3)
        print(len(word_set))
        transliterationLanguageGenerator.write_to_file(word_set, lang_output_path)