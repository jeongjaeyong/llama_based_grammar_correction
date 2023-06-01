from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import LlamaTokenizer
import pandas as pd
import torch
import loralib as lora
data = pd.read_csv("grammar_fixed.csv")
wrong_sentences = data['wrong_sentences'].to_list()
fixed_sentences = data['fixed_sentences'].to_list()
results = data['result'].to_list()


model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf").to(torch.bfloat16).to("cuda")
tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

prompt = "Hey, are you consciours? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
# generate_ids = model.generate(inputs.input_ids.to('cuda'), max_length=30)
generate_ids = model.generate(inputs.input_ids.to('cuda'), max_length=30)
return_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(return_text)

'''
train
'''

lora.mark_only_lora_as_trainable(model)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
epochs = 10
for epoch in range(epochs):
    for wrong_sentence, fixed_sentence, result in zip(wrong_sentences, fixed_sentences, results):
        prompt = 'Please Fix the this Sentence : "' + wrong_sentence + '" \n\nResult : "' + fixed_sentence +'"\n\nExplanation : ' + result
        inputs = tokenizer(prompt, return_tensors="pt")
        ouputs = model(inputs.input_ids.to('cuda'), labels = inputs.input_ids.to('cuda'))
        optimizer.zero_grad()
        ouputs.loss.backward()
        optimizer.step()

torch.save(lora.lora_state_dict(model), "llama2.pt")