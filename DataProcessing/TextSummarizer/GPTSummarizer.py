from openai import OpenAI
import os

os.environ["OPENAI_API_KEY"] = "<key>"

client = OpenAI(
  api_key=os.environ["OPENAI_API_KEY"]  # this is also the default, it can be omitted
)

# Replace YOUR_API_KEY with your actual API key
# openai.api_key = "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"

# The text you want to summarize
text = """There are two ways of extracting text using TextRank: keyword and sentence extraction. 
Keyword extraction can be done by simply using a frequency test, but this would almost always prove to be inaccurate. This is where TextRank automates the process to semantically provide far more accurate results based on the corpus.
Sentence extraction, on the other hand, studies the corpus to summarize the most valid sentences pertaining to the subject matter and phonologically arranges it. """

# The length of the summary you want, in number of words
summary_length = 30

# Use the summarize function to generate a summary of the text
# summary = client.chat.completions.create(
#   engine="davinci",
#   prompt=f"Summarize this text in {summary_length} words or fewer: {text}",
#   max_tokens=summary_length,
#   temperature=0,
#   top_p=1,
#   frequency_penalty=0,
#   presence_penalty=0
# ).text

completion = client.chat.completions.create(
        model='gpt-3.5-turbo', # newest, cheapest model
        messages=f"Summarize this text in {summary_length} words or fewer: {text}",
    )
summary = completion.choices[0].message.content

print(summary)