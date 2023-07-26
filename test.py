import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("wanderer2k1/gpt2-vietnamese-laws-qa")
model = GPT2LMHeadModel.from_pretrained("wanderer2k1/gpt2-vietnamese-laws-qa")

text = "Theo Luật Du Lịch, hợp đồng lữ hành không cần phải được lập thành văn bản. Đúng hay sai?"
input_ids = tokenizer.encode(text, return_tensors='pt')
max_length = 100

sample_outputs = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id,
                                do_sample=True,
                                max_length=max_length,
                                min_length=max_length,
                                top_k=40,
                                num_beams=5,
                                early_stopping=True,
                                no_repeat_ngram_size=2,
                                num_return_sequences=3)

for i, sample_output in enumerate(sample_outputs):
    print(">> Generated text {}\n\n{}".format(
        i+1, tokenizer.decode(sample_output.tolist())))
    print('\n---')
