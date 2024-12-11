import numpy as np
import sys,os
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
base_dir='./dataset/POPQURON/politeness_rating/Results/'
save_directory=base_dir+save_name+'/'
import time
import random
delay = random.uniform(0, 10)
time.sleep(delay)

if not os.path.exists(save_directory):
    os.makedirs(save_directory)
dataset=pd.read_csv('./dataset/POPQURON/politeness_rating/'+dataset_name+'.csv')
#prompt_name='Default'
with open('./dataset/POPQURON/politeness_rating/prompt_base.txt', 'r') as f:
    partial_prompt = f.read()

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

choice_to_score = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
dataset['gender'] = dataset['gender'].str.lower()
dataset['occupation']= dataset['occupation'].str.lower()
dataset['occupation'] = dataset['occupation'].replace('student', 'a student')
dataset['occupation'] = dataset['occupation'].replace('homemaker', 'a homemaker')
dataset['education'] = dataset['education'].str.lower()
dataset['education'] = dataset['education'].replace('college degree', 'a college degree')
dataset['education'] = dataset['education'].replace('graduate degree', 'a graduate degree')
dataset['education'] = dataset['education'].replace('high school diploma or equivalent', 'a high school diploma or equivalent')

# debug
#sample = dataset
#model = 'gpt-3.5-turbo'
def get_annotation(dataset,model):
    model_answer=[]
    probabilities=[]
    weighted_answer=[]
    for i in tqdm(range(len(dataset))):
        sample = dataset.iloc[i]
        if prompt_name == "No_SV":
            system_prompt=''
        else:
            system_prompt_map = {
                "Default": """**Your Profile**: In terms of race or ethnicity, you are %s. You are a %s. You are %s years old. Occupation-wise, you are %s. Your education level is %s. \n""" % (
                    sample['race'], sample['gender'], sample['age'],
                    sample['occupation'], sample['education'])

            }
            system_prompt = system_prompt_map[prompt_name]
        prompt = partial_prompt % sample['text']
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
