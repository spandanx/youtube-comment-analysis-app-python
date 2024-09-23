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
    # res = indicToEngTranslator.translate_sentence("বসন্ত")
    txts = [
        "Aap chor Bagan dekha nhi sayad...near mg metro.. one of finest pandal I bet",
        "Dada, ap north Kolkata ka, another ak big Puja visit kariyega, naw para dada vai sangha (baranagar). Ehaka thim hei &quot;পরিচয়&quot; Ake dekhiye mei geranty deta hu ki apko acha lage ga",
        "Coming to kolkata on 5th oct, kindly guide us which pandal to visit.. as its last day",
        "Chorbagan ta top 10 a rakhle valo hoto"
    ]
    for text in txts:
        res = indicToEngTranslator.translate_sentence(text)
        print(res)

