from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2Model.from_pretrained('gpt2-large')
text = "Replace me by any text you'd like o o o o o o o o o ."
encoded_input = tokenizer(text, return_tensors='pt')
print(type(encoded_input))
for k, v in encoded_input.items():
    print(k, type(v), v.shape)
output = model(**encoded_input)
print(type(output))
