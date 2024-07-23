from googletrans import Translator

class IndicToEngTranslator:

    def __init__(self):
        self.translator = Translator()

    def translate_sentence(self, sentence):
        res = self.translator.translate(sentence, "en")
        return res.text


if __name__ == "__main__":
    # res = translator.detect("sundor")
    # print(res)
    indicToEngTranslator = IndicToEngTranslator()
    # res = translator.translate("khub sundor hoy বসন্তের ভ্রমণ", "en")
    # print(indicToEngTranslator.translate_sentence("निर्माली বসন্তের ভ্রমণ ଦିବ୍ୟମୋହ"))
    res = indicToEngTranslator.translate_sentence("বসন্ত")
    print(res)

