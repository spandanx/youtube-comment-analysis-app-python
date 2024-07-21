from time import time
import enchant

from CharacterLanguageGeneraterDetecter.LanguageCharacterDetector import LanguageCharacterDetector
from IndicTransliteration.TransliterationIndicLatin2Native import TransliterationIndicLatin2Native
from LanguageTranslation.BengaliToEnglishTranslation import IndicToEngTranslator

d = enchant.Dict("en_US")
# import nltk
# nltk.download()
from nltk.corpus import words
nltk_words = set(words.words())
text_path = "C:\\Users\\spand\\OneDrive\\Documents\\Sample_english_text.txt"

class LanguageDetectorMain:

    def __init__(self):
        self.trans_supported_langs = set(
            ["ben", "guj", "hin", "kan", "mal", "mar", "nep", "pan", "ori", "san", "tam", "tel", "urd"])

    def transliterate_word(self, word_desc_map, word, trans_object):
        meaning_lang = word_desc_map["meaning_lang"]
        if meaning_lang in self.trans_supported_langs:
            transed_word = trans_object.transliterate_to_native(word, meaning_lang)
            # print(transed_word)
            word_desc_map.update({"transliterated_word": transed_word})
        return word_desc_map


    def detect_language_of_text(self, lines, languageCharacterDetector, transliteration_object, language_translation_object):
        # nltk_wordset = set()
        # enchant_wordset = set()
        lang_list = []
        for line in lines:
            print(line)
            word_list = []
            words = line.split(' ')
            for word in words:
                print(word)
                if len(word)==0:
                    continue
                current_word = {
                    "word": word
                }
                if d.check(word):
                    print("Present - ", word)
                    current_word.update({"letter_lang": "english", "meaning_lang": "english", "confidence": 100.0})
                else:
                    current_word.update(languageCharacterDetector.detect_word_lang(word))
                    if 'meaning_lang' in current_word:
                        current_word = self.transliterate_word(current_word, current_word["word"], transliteration_object)
                        # current_word = transliteration_object.transliterate_word(current_word, current_word["word"])
                        if 'transliterated_word' in current_word:
                            transliterated_word = current_word["transliterated_word"]
                            translated_word = language_translation_object.translate_word(transliterated_word)
                            current_word.update({"translated_word": translated_word})
                    if ('letter_lang' in current_word) and (current_word['letter_lang']!='eng'):
                        translated_word = language_translation_object.translate_word(current_word['word'])
                        current_word.update({"translated_word": translated_word})



                word_list.append(current_word)
            lang_list.append(word_list)
        return lang_list

        # if string in nltk_words:
        #     nltk_wordset.add(string)
if __name__ == "__main__":
    file = open(text_path, "r")
    content = file.read()
    file.close()

    lines = content.split('\n')

    t = time()

    lines = [
        # "are more simply constructed from the 7-orthoplex.",
        "are farmers, while an additional 5% receives their livelihood from raising livestock.",
        "বসন্তের ভ্রমণ निर्माली"
        # "nirmaali shivaalapurva siddhanto"
    ]

    languageDetector = LanguageDetectorMain()
    languageCharacterDetector = LanguageCharacterDetector()
    # lcd = LanguageCharacterDetector()
    tl = TransliterationIndicLatin2Native()
    indicToEngTranslator = IndicToEngTranslator()
    lang_list = languageDetector.detect_language_of_text(lines, languageCharacterDetector, tl, indicToEngTranslator)
    t1 = time()
    print(t1 - t)
    for line in lang_list:
        print(line)
# for wd in enchant_wordset:
#     if wd not in nltk_wordset:
#         print(wd)