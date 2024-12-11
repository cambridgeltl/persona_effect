import numpy as np
import sys,os
import openai
import pickle
import pandas as pd
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

base_dir='./dataset/AnnotatorWithAttitudes/Results/'
save_directory=base_dir+save_name+'/'
import time
import random
delay = random.uniform(0, 10)
time.sleep(delay)

if not os.path.exists(save_directory):
    os.makedirs(save_directory)
dataset=pd.read_csv('./dataset/AnnotatorWithAttitudes/'+dataset_name+'.csv')
with open('./dataset/AnnotatorWithAttitudes/prompt_base.txt', 'r') as f:
    partial_prompt = f.read()

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

choice_to_score = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}

def get_annotation(dataset,model):
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
    model_answer=[]
    probabilities=[]
    weighted_answer=[]

    for i in tqdm(range(len(dataset))):
        sample = dataset.iloc[i]

        if prompt_name == "No_SV":
            system_prompt=''
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
        prompt = partial_prompt % sample['tweet']
        messages = []
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        chat_completion = call_model_with_retry(messages,model)
        raw_negative_log_prob=chat_completion['choices'][0]['logprobs']['content'][0]['top_logprobs']
        choice_order=[]
        prob=[]
        for i1 in range(len(raw_negative_log_prob)):
            choice_order.append(raw_negative_log_prob[i1]['token'])
            prob.append(raw_negative_log_prob[i1]['logprob'])
        prob=softmax(prob)
        prob_dict = dict(zip(choice_order, prob))

        # Sort the dictionary by key (which will be in the order ABCDE)
        sorted_prob_dict = dict(sorted(prob_dict.items()))

        # Extract the values in the sorted order
        sorted_prob = list(sorted_prob_dict.values())
        probabilities.append(sorted_prob)
        model_answer.append(choice_order[0])
        scores = list(choice_to_score.values())
        weighted_answer.append(np.dot(np.array(scores),sorted_prob))
        print(messages)
        print(choice_order[0])
    return model_answer,weighted_answer,probabilities

model_answer,weighted_answer,scores = get_annotation(dataset,model_name)

# save all three results
with open(save_directory+dataset_name+prompt_name+'.pkl', 'wb') as handle:
    pickle.dump([model_answer,weighted_answer,scores], handle, protocol=pickle.HIGHEST_PROTOCOL)


