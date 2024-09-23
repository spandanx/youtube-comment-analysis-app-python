import pickle

if __name__ == "__main__":
    char_file_path = "./data/transliteration_words/"
else:
    char_file_path = "./data/transliteration_words/"



class TransliterationLanguageDetector:

    def __init__(self):
        self.word_maps = dict()
        self.langs = ["ben", "guj", "hin", "kan", "mal", "mar", "nep", "pan", "ori", "san", "tam", "tel", "urd"]
        for lang in self.langs:
            lang_output_path = char_file_path + 'transliteration_words_' + lang + '.txt'
            self.load_char_maps(lang, lang_output_path)

    def load_char_maps(self, lang, char_file_path):
        with open(char_file_path, 'rb') as f:
            my_set = pickle.load(f)
        self.word_maps[lang] = my_set

    def detect_word_lang(self, word):
        if len(word) == 0:
            return {"letter_lang": "EMPTY", "meaning_lang": "EMPTY", "confidence": 100.0}
        for key, value in self.word_maps.items():
            if word in value:
                return {"letter_lang": 'eng', "meaning_lang": key, "confidence": 100.0}
        return {"letter_lang": "eng", "meaning_lang": "NOT_FOUND", "confidence": 100.0}

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
    transliterationLanguageDetector = TransliterationLanguageDetector()
    # for lang in lang_array:
    #     lang_output_path = char_file_path + lang + '_char.txt'
    #     otherLanguageCharacterDetector.load_char_maps(lang, lang_output_path)
    # print(otherLanguageCharacterDetector.word_maps)
    word = "chalo jaoya jak"
    print(transliterationLanguageDetector.detect_sentence_lang(word))