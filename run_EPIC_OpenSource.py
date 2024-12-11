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
base_dir='./dataset/EPIC/Results/'
save_directory=base_dir+save_name+'/'

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

dataset = pd.read_csv(f'./dataset/EPIC/{dataset_name}.csv')
with open('./dataset/EPIC/prompt_base.txt', 'r') as f:
    partial_prompt = f.read()
partial_prompt = partial_prompt[:-14]


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

choice_to_score = {'A': 1, 'B': 0}
def prepare_data(sample, prompt_name):
    gender_map = {item: item.lower() for item in dataset['Sex']}
    are_arenot_map = {'Yes': 'are', 'No': 'are not'}
    employment_map = {"Full-Time":"full-time employed", "Part-Time":"part-time employed", "Unemployed (and job seeking)":"unemployed and job-seeking", "Not in paid work (e.g. homemaker', 'retired or disabled)":"Not in paid work (e.g. homemaker', 'retired or disabled)"}
    instruction_prompt = partial_prompt % (sample['parent_text'],sample['text'])

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
            "Default": """**Your Profile**: You ethnicity is %s. Your gender is %s. You are %s years old. """ \
                       """Your country of birth is %s. Your country of residence is %s. You are a national of %s. """\
                       """You %s a student. You are %s."""

                       % (sample['Ethnicity.simplified'], gender_map[sample['Sex']], sample['Age'],
                           sample['Country.of.birth'], sample['Country.of.residence'],sample['Nationality'],
                           are_arenot_map[sample['Student.status']], employment_map[sample['Employment.status']]),

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

BATCH_SIZE = batch_size
num_batches = len(dataset) // BATCH_SIZE
total_final_answers=[]
total_weighted_final_answers=[]
total_probabilities=[]

for batch_idx in tqdm(range(num_batches)):
    start_idx = batch_idx * batch_size
    end_idx = (batch_idx + 1) * batch_size
    batch_data = [prepare_data(dataset.iloc[i], prompt_name) for i in range(start_idx, end_idx)]
    input_ids = tokenizer(batch_data, return_tensors="pt", padding=True).to('cuda')

    output_ids = model.generate(**input_ids, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id,
                                max_new_tokens=1, output_scores=True, return_dict_in_generate=True, renormalize_logits=True)

    desired_tokens = list(choice_to_score.keys())
    scores = list(choice_to_score.values())
    desired_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in desired_tokens]

    final_answers = []
    weighted_final_answers = []
    probabilities = []

    for idx in range(output_ids['scores'][0].shape[0]):
        logits_for_desired_tokens = output_ids['scores'][idx][0, desired_token_ids].cpu().detach().numpy()
        final_answer = desired_tokens[np.argmax(logits_for_desired_tokens)]
        final_answers.append(choice_to_score[final_answer])
        logits_for_desired_tokens[logits_for_desired_tokens == -np.inf] = -np.finfo(np.float32).max
        probabilities_from_logits = softmax(logits_for_desired_tokens)
        weighted_final_answer = np.dot(np.array(scores), probabilities_from_logits)
        weighted_final_answers.append(weighted_final_answer)
        probabilities.append(probabilities_from_logits)

    total_final_answers.extend(final_answers)
    total_weighted_final_answers.extend(weighted_final_answers)
    total_probabilities.extend(probabilities)

# Save results
with open(os.path.join(save_directory, f'{dataset_name}{prompt_name}.pkl'), 'wb') as handle:
    pickle.dump([total_final_answers, total_weighted_final_answers, total_probabilities], handle, protocol=pickle.HIGHEST_PROTOCOL)