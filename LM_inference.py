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
model.load_state_dict(torch.load('llama2.pt'), strict=False)



# Generate
# generate_ids = model.generate(inputs.input_ids.to('cuda'), max_length=30)
for wrong_sentence, fixed_sentence, result in zip(wrong_sentences, fixed_sentences, results):
    prompt = 'Please Fix the this Sentence : "' + "I has cat" + '" \n\nResult : "'
    inputs = tokenizer(prompt, return_tensors="pt")

    generate_ids = model.generate(inputs.input_ids.to('cuda'), max_length=256)
    return_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(return_text)
    break
