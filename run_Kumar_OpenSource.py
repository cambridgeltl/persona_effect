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
base_dir='./dataset/Kumar/Results/'
save_directory=base_dir+save_name+'/'

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

dataset=pd.read_csv('./dataset/Kumar/'+dataset_name+'.csv')
with open('./dataset/Kumar/prompt_base.txt', 'r') as f:
    partial_prompt = f.read()
partial_prompt = partial_prompt[:-14]

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

choice_to_score = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'E': 0}
def prepare_data(sample, prompt_name):
    race_map = {item: item.replace(',', ' and ') for item in dataset['race']}
    gender_map = {item: item.lower() for item in dataset['gender']}
    do_donot_map={'Yes':'do','No':'do not'}
    lgbtq_map = {item: item.lower() for item in dataset['lgbtq_status']}
    religion_map = {item: item.lower() for item in dataset['religion_important']}
    political_map = {item: item.lower() for item in dataset['political_affilation']}
    are_arenot_map = {'Yes': 'are', 'No': 'are not'}
    do_donot_map_2={True:'do',False:'do not'}
    tech_impact_map={item: item.lower() for item in dataset['technology_impact']}
    toxicpost_map={item: item.lower() for item in dataset['toxic_comments_problem']}
    instruction_prompt = partial_prompt % sample['comment']

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
            "Default": """**Your Profile**: You race is %s. Your gender is %s. You %s describe yourself as transgender. You consider yourself to be %s. You are %s years old. """ \
                       """The highest degree or level of school that you have completed is %s. Religion is %s in your life. Generally speaking, you consider yourself politically %s. """\
                       """You %s a parent. You %s use social networking sites (Facebook, Twitter). You %s use video sites (YouTube, Twitch). """ \
                       """You %s use news sites (CNN, Fox, NYT, WSJ). You %s use community forums (Reddit, Craigslist, 4chan). In general, you rate the impact of technology on people’s lives as %s. """\
                       """Based on your experience, toxic posts or comments are %s."""

                       % (race_map[sample['race']], gender_map[sample['gender']], do_donot_map[sample['identify_as_transgender']],lgbtq_map[sample['lgbtq_status']], sample['age_range'],
                           sample['education'], religion_map[sample['religion_important']],political_map[sample['political_affilation']],
                           are_arenot_map[sample['is_parent']],do_donot_map_2[sample['uses_media_social']],do_donot_map_2[sample['uses_media_video']],
                           do_donot_map_2[sample['uses_media_news']], do_donot_map_2[sample['uses_media_forums']], tech_impact_map[sample['technology_impact']],
                           toxicpost_map[sample['toxic_comments_problem']]),
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
    print(batch_data)
    input_ids = tokenizer(batch_data, return_tensors="pt", padding=True)
    input_ids = input_ids.to('cuda')
    output_ids = model.generate(**input_ids, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id,
                            max_new_tokens=1, output_scores=True, return_dict_in_generate=True, renormalize_logits=True)

    desired_tokens = list(choice_to_score.keys())
    scores = list(choice_to_score.values())
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


