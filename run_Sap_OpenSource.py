import os
import numpy as np
import pandas as pd
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

# add arguments to the parser
parser.add_argument('--model', type=str, help='model name')
parser.add_argument('--batch_size', type=str, help='batch size')
parser.add_argument('--dataset_name',type=str,help='dataset name')
parser.add_argument('--prompt_name',type=str,help='prompt name')
# parse the arguments
args = parser.parse_args()
# access the values of the arguments
model_name = args.model
batch_size= int(args.batch_size)
dataset_name=args.dataset_name
prompt_name=args.prompt_name

if ('t5' in model_name) or ('flan' in model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map='auto', load_in_4bit=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

elif ('Llama' in model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', load_in_4bit=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

else:
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', load_in_4bit=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = model.config.eos_token_id
model.eval()

if '/' in model_name:
    save_name = model_name.split('/')[1]
else:
    save_name = model_name


pd.set_option('display.max_rows', 200)

"""
Start with small scale dataset and see
"""
base_dir='./dataset/AnnotatorWithAttitudes/Results/'
save_directory=base_dir+save_name+'/'

if not os.path.exists(save_directory):
    os.makedirs(save_directory)


dataset=pd.read_csv('./dataset/AnnotatorWithAttitudes/'+dataset_name+'.csv')

with open('./dataset/AnnotatorWithAttitudes/prompt_base.txt', 'r') as f:
    partial_prompt = f.read()
partial_prompt=partial_prompt[:-14]
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

choice_to_score = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}

def prepare_data(sample, prompt_name):
    race_map = {'white': 'White/Caucasian', 'hisp': 'Hispanic/Latinx', 'black': 'Black/African American',
                'middleEastern': 'Middle Eastern', 'native': 'Native American/First Nations'}
    gender_map = {'man': 'man/male', 'woman': 'woman/female', 'nonBinary': 'non-binary'}
    age_map = {
        18: "18-24",
        25: "25-29",
        30: "30-34",
        35: "35-39",
        40: "40-44",
        45: "45-49",
        50: "50-54",
        55: "55-59",
        60: "60-64",
        65: "65+",
    }
    instruction_prompt = partial_prompt % sample['tweet']
    if prompt_name == "No_SV":
        if ("llama-2" in model_name.lower()) and ("chat" not in model_name.lower()):
            messages = instruction_prompt + '\n**Answer**: ('
        if ("llama-2" in model_name.lower()) and ("chat" in model_name.lower()):
            messages = """<s>[INST]{user_message} [/INST]\n**Answer**: (""".format(user_message=instruction_prompt)
        if "tulu" in model_name:
            messages = """<|user|>\n{user_message}\n<|assistant|>**Answer**: (""".format(
                user_message=instruction_prompt)
        return messages
    else:
        system_prompt_map = {
            "Default": """**Your Profile**: You race/ethnicity is %s. Your gender identity is %s. You are %s years old. """ \
                       """In terms of your political orientation, on a five-point Likert scale of -1 (Strongly liberal) to 1 (Strongly conservative), you are %s."""
                       % (
                           race_map[sample['annotatorRace']], gender_map[sample['annotatorGender']],
                           age_map[sample['annotatorAge']],
                           sample['annotatorPolitics'])
        }

        system_prompt = system_prompt_map[prompt_name]
        if ("llama-2" in model_name.lower()) and ("chat" not in model_name.lower()):
            messages = system_prompt + '\n' + instruction_prompt + '\n**Answer**: ('
        if ("llama-2" in model_name.lower()) and ("chat" in model_name.lower()):
            messages = """<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message} [/INST]\n**Answer**: (""".format(
                system_prompt=system_prompt,
                user_message=instruction_prompt)
        if "tulu" in model_name:
            system_prompt = system_prompt + '\n' + instruction_prompt
            messages = """<|user|>\n{user_message}\n<|assistant|>**Answer**: (""".format(user_message=system_prompt)
    return messages

BATCH_SIZE = batch_size  # or whatever size fits in your memory
num_batches = len(dataset) // BATCH_SIZE
total_final_answers=[]
total_weighted_final_answers=[]
total_probabilities=[]

for batch_idx in tqdm(range(num_batches)):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = (batch_idx + 1) * BATCH_SIZE
    # Collect batched data
    batch_data = [prepare_data(dataset.iloc[i],prompt_name) for i in
                  range(start_idx, end_idx)]  # assuming prepare_data gives the required format for each sample
    input_ids = tokenizer(batch_data, return_tensors="pt", padding=True)
    input_ids = input_ids.to('cuda')
    output_ids = model.generate(**input_ids, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id,
                            max_new_tokens=1, output_scores=True, return_dict_in_generate=True, renormalize_logits=True)

    desired_tokens = ["A", "B", "C", "D", "E"]
    scores = [1,2,3,4,5]
    desired_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in desired_tokens]
    final_answers=[]
    weighted_final_answers=[]
    probabilities=[]
    #Loop through each batch element
    for idx in range(output_ids['scores'][0].shape[0]):
        # Get the logits for the desired tokens for each batch element
        logits_for_desired_tokens = output_ids['scores'][idx][0, desired_token_ids]
        logits_for_desired_tokens = logits_for_desired_tokens.cpu().detach().numpy()
        # Find the token with the maximum logit
        final_answer = desired_tokens[np.argmax(logits_for_desired_tokens)]
        final_answers.append(choice_to_score[final_answer])
        print("logits",logits_for_desired_tokens)
        print("final answer",final_answers)
        logits_for_desired_tokens[logits_for_desired_tokens == -np.inf] = -np.finfo(np.float32).max
        # Calculate probabilities using softmax
        probabilities_from_logits = softmax(logits_for_desired_tokens)
        weighted_final_answer=np.dot(np.array(scores),probabilities_from_logits)
        weighted_final_answers.append(weighted_final_answer)
        probabilities.append(probabilities_from_logits)
        print("weighted",weighted_final_answer)
    total_final_answers.extend(final_answers)
    total_weighted_final_answers.extend(weighted_final_answers)
    total_probabilities.extend(probabilities)

with open(save_directory+dataset_name+prompt_name+'.pkl', 'wb') as handle:
    pickle.dump([total_final_answers,total_weighted_final_answers,total_probabilities], handle, protocol=pickle.HIGHEST_PROTOCOL)


