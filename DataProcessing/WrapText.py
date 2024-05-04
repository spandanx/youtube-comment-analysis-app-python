import textwrap

def wrapText(textArray):
    final_text = ''
    for text in textArray:
        final_text = final_text + textwrap.fill(text, replace_whitespace=False, fix_sentence_endings=True) + "\n"
    if final_text:
        final_text = final_text[:-1] # removing final new line
    return final_text

if __name__ == "__main__":
    textArray = [
        "About 55,000 years ago, the first modern humans, or Homo sapiens, had arrived on the Indian subcontinent from Africa, where they had earlier evolved.",
        "The earliest known modern human remains in South Asia date to about 30,000 years ago.",
        "After 6500 BCE, evidence for domestication of food crops and animals, construction of permanent structures, and storage of agricultural surplus appeared in Mehrgarh and other sites in Balochistan, Pakistan. These gradually developed into the Indus Valley Civilisation, the first urban culture in South Asia, which flourished during 2500–1900 BCE in modern day Pakistan and western India. Centred around cities such as Mohenjo-daro, Harappa, Dholavira, and Kalibangan, and relying on varied forms of subsistence, the civilisation engaged robustly in crafts production and wide-ranging trade. Ancient India was one of the four Old World cradles of civilization."
    ]
    result = wrapText(textArray)
    a = 1
