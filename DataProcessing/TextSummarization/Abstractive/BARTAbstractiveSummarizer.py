from transformers import BartTokenizer, BartForConditionalGeneration


class BARTAbstractiveSummarizer:

    def __init__(self):
        model_name = "facebook/bart-large-cnn"
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)

    def summarizeText(self, text):
        # inputs = self.tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        inputs = self.tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self.model.generate(inputs, max_length=100, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = ""
        for summary_id in summary_ids:
            local_summary = self.tokenizer.decode(summary_id, skip_special_tokens=True)
            summary += local_summary
        return summary

if __name__ == "__main__":
    text = """
    There are two ways of extracting text using TextRank: keyword and sentence extraction. 
    Keyword extraction can be done by simply using a frequency test, but this would almost always prove to be inaccurate. This is where TextRank automates the process to semantically provide far more accurate results based on the corpus.
    Sentence extraction, on the other hand, studies the corpus to summarize the most valid sentences pertaining to the subject matter and phonologically arranges it. 
    """
    bartAbstractiveSummarizer = BARTAbstractiveSummarizer()
    print(bartAbstractiveSummarizer.summarizeText(text))