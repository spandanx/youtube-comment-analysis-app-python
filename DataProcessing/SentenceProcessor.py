from CharacterLanguageGeneraterDetecter.LanguageCharacterDetector import LanguageCharacterDetector
from IndicTransliteration.TransliterationIndicLatin2Native import TransliterationIndicLatin2Native

if __name__ == "__main__":
    languageCharacterDetector = LanguageCharacterDetector()
    tl = TransliterationIndicLatin2Native()
    languageCharacterDetector.set_transliteration_object(tl)
    # for lang in lang_array:
    #     lang_output_path = char_file_path + lang + '_char.txt'
    #     otherLanguageCharacterDetector.load_char_maps(lang, lang_output_path)
    print(languageCharacterDetector.char_maps)
    word = "siddhanto বসন্তের ভ্রমণ निर्माली there"
    print(languageCharacterDetector.detect_sentence_lang(word))