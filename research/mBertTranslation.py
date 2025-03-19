from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, MBart50Tokenizer

article_bn = "কলকাতা জাতীয় গুরুত্বপূর্ণ প্রতিষ্ঠানের আবাসস্থল"
article_hi = "संयुक्त राष्ट्र के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है"
article_ar = "الأمين العام للأمم المتحدة يقول إنه لا يوجد حل عسكري في سوريا."

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")


tokenizer.src_lang = "bn_IN"
encoded_bn = tokenizer(article_bn, return_tensors="pt")
generated_tokens = model.generate(
    **encoded_bn,
    # forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"]
    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
)
res = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(res)

# translate Hindi to French
tokenizer.src_lang = "hi_IN"
encoded_hi = tokenizer(article_hi, return_tensors="pt")
generated_tokens = model.generate(
    **encoded_hi,
    # forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"]
    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
)
res = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(res)
# => "Le chef de l 'ONU affirme qu 'il n 'y a pas de solution militaire dans la Syrie."

# translate Arabic to English
tokenizer.src_lang = "ar_AR"
encoded_ar = tokenizer(article_ar, return_tensors="pt")
generated_tokens = model.generate(
    **encoded_ar,
    # forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
)
res = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(res)
# => "The Secretary-General of the United Nations says there is no military solution in Syria."
