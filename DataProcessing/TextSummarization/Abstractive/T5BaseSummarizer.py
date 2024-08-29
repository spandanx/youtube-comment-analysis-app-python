from transformers import AutoTokenizer, AutoModelWithLMHead

class T5BaseSummarizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('t5-base')
        self.model = AutoModelWithLMHead.from_pretrained('t5-base', return_dict=True)

    def summarizeText(self, text):
        inputs = self.tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=512, truncation=True)
        summary_ids = self.model.generate(inputs, max_length=150, min_length=80, length_penalty=5., num_beams=2)

        summary = ""
        for summary_id in summary_ids:
            local_summary = self.tokenizer.decode(summary_id)
            local_summary = local_summary.replace("<pad> ", "").replace("</s>", "")
            summary += local_summary
        return summary


if __name__ == "__main__":
    text = """
    There are two ways of extracting text using TextRank: keyword and sentence extraction. 
    Keyword extraction can be done by simply using a frequency test, but this would almost always prove to be inaccurate. This is where TextRank automates the process to semantically provide far more accurate results based on the corpus.
    Sentence extraction, on the other hand, studies the corpus to summarize the most valid sentences pertaining to the subject matter and phonologically arranges it. 
    """
    t5BaseSummarizer = T5BaseSummarizer()
    print(t5BaseSummarizer.summarizeText(text))