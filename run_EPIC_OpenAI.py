import pandas as pd
import numpy as np
import sys,os
sys.path.append(os.getcwd())

import openai
import pickle
import pandas as pd
# print the first model's id
from tqdm import tqdm
from Dependency import call_model_with_retry,set_seed
import signal
pd.set_option('display.max_rows', 200)
set_seed(42)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='model name')
parser.add_argument('--dataset_name',type=str,help='dataset name')
parser.add_argument('--prompt_name',type=str,help='prompt name')
args = parser.parse_args()
model_name = args.model
dataset_name=args.dataset_name
prompt_name=args.prompt_name

if '/' in model_name:
    save_name = model_name.split('/')[1]
else:
    save_name = model_name

base_dir='./dataset/EPIC/Results/'
save_directory=base_dir+save_name+'/'
import time
import random
#delay = random.uniform(0, 10)
#time.sleep(delay)

if not os.path.exists(save_directory):
    os.makedirs(save_directory)
dataset = pd.read_csv(f'./dataset/EPIC/{dataset_name}.csv')
#prompt_name='Default'
with open('./dataset/EPIC/prompt_base.txt', 'r') as f:
    partial_prompt = f.read()
# Do some simple preprocessing

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

choice_to_score = {'A': 1, 'B': 0}

# Function to get annotation
def get_annotation(dataset, model):
    gender_map = {item: item.lower() for item in dataset['Sex']}
    are_arenot_map = {'Yes': 'are', 'No': 'are not'}
    employment_map = {
        "Full-Time": "full-time employed",
        "Part-Time": "part-time employed",
        "Unemployed (and job seeking)": "unemployed and job-seeking",
        "Not in paid work (e.g. homemaker', 'retired or disabled)": "Not in paid work (e.g. homemaker', 'retired or disabled)"
    }
    model_answer = []
    probabilities = []
    weighted_answer = []

    for i in tqdm(range(len(dataset))):
        sample = dataset.iloc[i]

        if prompt_name == "No_SV":
            system_prompt = ''
        else:
            system_prompt = f"""**Your Profile**: You ethnicity is {sample['Ethnicity.simplified']}. Your gender is {gender_map[sample['Sex']]}. You are {sample['Age']} years old. """ \
                            f"""Your country of birth is {sample['Country.of.birth']}. Your country of residence is {sample['Country.of.residence']}. You are a national of {sample['Nationality']}. """ \
                            f"""You {are_arenot_map[sample['Student.status']]} a student. You are {employment_map[sample['Employment.status']]}."""

        prompt = partial_prompt % (sample['parent_text'], sample['text'])
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        chat_completion = call_model_with_retry(messages, model)
        raw_negative_log_prob = chat_completion['choices'][0]['logprobs']['content'][0]['top_logprobs']
        choice_order = []
        prob = []

        for i1 in range(len(raw_negative_log_prob)):
            choice_order.append(raw_negative_log_prob[i1]['token'])
            prob.append(raw_negative_log_prob[i1]['logprob'])

        prob = softmax(prob)
        prob_dict = dict(zip(choice_order, prob))

        # Sort the dictionary by key (which will be in the order ABCDE)
        sorted_prob_dict = dict(sorted(prob_dict.items()))

        # Extract the values in the sorted order
        sorted_prob = list(sorted_prob_dict.values())[:len(choice_to_score)]
        probabilities.append(sorted_prob)
        model_answer.append(choice_order[0])
        scores = list(choice_to_score.values())
        weighted_answer.append(np.dot(np.array(scores), sorted_prob))

    return model_answer, weighted_answer, probabilities

# Get annotations
model_answer, weighted_answer, scores = get_annotation(dataset, model_name)

# Save results
with open(os.path.join(save_directory, f'{dataset_name}{prompt_name}.pkl'), 'wb') as handle:
    pickle.dump([model_answer, weighted_answer, scores], handle, protocol=pickle.HIGHEST_PROTOCOL)