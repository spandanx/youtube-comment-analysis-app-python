from time import time
import enchant

from CharacterLanguageGeneraterDetecter.LanguageCharacterDetector import LanguageCharacterDetector
from IndicTransliterationStandBy.TransliterationIndicLatin2Native import TransliterationIndicLatin2Native
from LanguageTranslation.IndicToEnglishTranslation import IndicToEngTranslator

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
        self.language_character_detector = LanguageCharacterDetector()
        self.transliteration_object = TransliterationIndicLatin2Native()
        self.language_translation_object = IndicToEngTranslator()

    def transliterate_word(self, word_desc_map, word, trans_object):
        meaning_lang = word_desc_map["meaning_lang"]
        if meaning_lang in self.trans_supported_langs:
            transed_word = trans_object.transliterate_to_native(word, meaning_lang)
            # print(transed_word)
            word_desc_map.update({"transliterated_word": transed_word})
        return word_desc_map

    def convert_language_of_text(self, line):
        indic_words_native = ""
        indic_words_eng = ""
        output_words = ""
        word_list = []
        words = line.split(' ')
        for word in words:
            # print(word)
            if len(word)==0:
                continue
            current_word = {
                "word": word
            }
            current_word.update(self.language_character_detector.detect_word_lang(word))
            if (current_word["letter_lang"] == "eng"):
                if (len(indic_words_native) > 0):
                    output_words += self.language_translation_object.translate_sentence(indic_words_native) + " "
                    indic_words_native = ""
                # indic_words_eng += word + " "
                output_words += word + " "
            else:
                # if (len(indic_words_eng) > 0):
                #     output_words += self.language_translation_object.translate_sentence(indic_words_eng) + " "
                #     indic_words_eng = ""
                indic_words_native += word + " "
            # indic_words += word + " "

            word_list.append(current_word)
        # print(word_list)

        if (len(indic_words_native) > 0):
            output_words += self.language_translation_object.translate_sentence(indic_words_native) + " "
            # indic_words_native = ""
        # if (len(indic_words_eng) > 0):
        #     output_words += self.language_translation_object.translate_sentence(indic_words_eng) + " "
            # indic_words_eng = ""

        if (len(output_words) > 0 and output_words[-1] == ' '):
            output_words = output_words[:-1]
        output_words_res = self.language_translation_object.translate_sentence(output_words)
        return output_words_res

    def detect_language_of_text(self, line):
        # nltk_wordset = set()
        # enchant_wordset = set()
        # lang_list = []
        # for line in lines:
        # print(line)
        indic_words_native = ""
        indic_words_eng = ""
        output_words = ""
        word_list = []
        words = line.split(' ')
        for word in words:
            # print(word)
            if len(word)==0:
                continue
            current_word = {
                "word": word
            }
            if d.check(word):
                # print("Present - ", word)
                current_word.update({"letter_lang": "eng", "meaning_lang": "eng", "confidence": 100.0})
                if (len(indic_words_native) > 0):
                    output_words += self.language_translation_object.translate_sentence(indic_words_native) + " "
                    indic_words_native = ""
                if (len(indic_words_eng) > 0):
                    output_words += self.language_translation_object.translate_sentence(indic_words_eng) + " "
                    indic_words_eng = ""
                output_words += word + " "
            else:
                current_word.update(self.language_character_detector.detect_word_lang(word))
                if (current_word["letter_lang"] == "eng"):
                    if (len(indic_words_native) > 0):
                        output_words += self.language_translation_object.translate_sentence(indic_words_native) + " "
                        indic_words_native = ""
                    indic_words_eng += word + " "
                else:
                    if (len(indic_words_eng) > 0):
                        output_words += self.language_translation_object.translate_sentence(indic_words_eng) + " "
                        indic_words_eng = ""
                    indic_words_native += word + " "
                # indic_words += word + " "

            word_list.append(current_word)
        # print(word_list)

        if (len(indic_words_native) > 0):
            output_words += self.language_translation_object.translate_sentence(indic_words_native) + " "
            # indic_words_native = ""
        if (len(indic_words_eng) > 0):
            output_words += self.language_translation_object.translate_sentence(indic_words_eng) + " "
            # indic_words_eng = ""

        if (len(output_words) > 0 and output_words[-1] == ' '):
            output_words = output_words[:-1]
                # if 'meaning_lang' in current_word:
                #     current_word = self.transliterate_word(current_word, current_word["word"], transliteration_object)
                #     # current_word = transliteration_object.transliterate_word(current_word, current_word["word"])
                #     if 'transliterated_word' in current_word:
                #         transliterated_word = current_word["transliterated_word"]
                #         translated_word = language_translation_object.translate_word(transliterated_word)
                #         current_word.update({"translated_word": translated_word})
                # if ('letter_lang' in current_word) and (current_word['letter_lang']!='eng'):
                #     translated_word = language_translation_object.translate_word(current_word['word'])
                #     current_word.update({"translated_word": translated_word})


        return output_words
        # lang_list.append(word_list)
        # lang_list.append(output_words)
        # return lang_list

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
        # "are farmers, while an additional 5% receives their livelihood from raising livestock.",
        # "বসন্তের ভ্রমণ निर्माली",
        # "khub sundor",
        # "বসন্তের journey khub sundor",
        # "journey was very sundor of বসন্ত"

            "Aap chor Bagan dekha nhi sayad...near mg metro.. one of finest pandal I bet",
            "Dada, ap north Kolkata ka, another ak big Puja visit kariyega, naw para dada vai sangha (baranagar). Ehaka thim hei &quot;পরিচয়&quot; Ake dekhiye mei geranty deta hu ki apko acha lage ga",
            "Coming to kolkata on 5th oct, kindly guide us which pandal to visit.. as its last day",
            "Chorbagan ta top 10 a rakhle valo hoto"
        # "nirmaali shivaalapurva siddhanto"
    ]

    languageDetector = LanguageDetectorMain()
    languageCharacterDetector = LanguageCharacterDetector()
    # lcd = LanguageCharacterDetector()
    tl = TransliterationIndicLatin2Native()
    indicToEngTranslator = IndicToEngTranslator()
    for line in lines:
        print("Source - ")
        print(line)
        lang_list = languageDetector.detect_language_of_text(line)
        print("\t OLD - ")
        print("\t", lang_list)

        lang_list = languageDetector.convert_language_of_text(line)
        print("\t NEW - ")
        print("\t", lang_list)
    t1 = time()
    print(t1 - t)
    # for line in lang_list:
    #     print(line)
# for wd in enchant_wordset:
#     if wd not in nltk_wordset:
#         print(wd)