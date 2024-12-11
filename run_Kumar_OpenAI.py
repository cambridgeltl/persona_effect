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

base_dir='./dataset/Kumar/Results/'
save_directory=base_dir+save_name+'/'
import time
import random
delay = random.uniform(0, 10)
time.sleep(delay)

if not os.path.exists(save_directory):
    os.makedirs(save_directory)
dataset=pd.read_csv('./dataset/Kumar/'+dataset_name+'.csv')
#prompt_name='Default'
with open('./dataset/Kumar/prompt_base.txt', 'r') as f:
    partial_prompt = f.read()
# Do some simple preprocessing

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

choice_to_score = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'E': 0}

# debug
#sample = dataset
#model = 'gpt-3.5-turbo'
def get_annotation(dataset,model):
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
    model_answer=[]
    probabilities=[]
    weighted_answer=[]
    for i in tqdm(range(len(dataset))):
        sample = dataset.iloc[i]

        if prompt_name == "No_SV":
            system_prompt=''
        else:

            system_prompt_map = {
                "Default": """**Your Profile**: You race is %s. Your gender is %s. You %s describe yourself as transgender. You consider yourself to be %s. You are %s years old. """ \
                           """The highest degree or level of school that you have completed is %s. Religion is %s in your life. Generally speaking, you consider yourself politically %s. """ \
                           """You %s a parent. You %s use social networking sites (Facebook, Twitter). You %s use video sites (YouTube, Twitch). """ \
                           """You %s use news sites (CNN, Fox, NYT, WSJ). You %s use community forums (Reddit, Craigslist, 4chan). In general, you rate the impact of technology on people’s lives as %s. """ \
                           """Based on your experience, toxic posts or comments are %s."""

                           % (race_map[sample['race']], gender_map[sample['gender']],
                              do_donot_map[sample['identify_as_transgender']], lgbtq_map[sample['lgbtq_status']],
                              sample['age_range'],
                              sample['education'], religion_map[sample['religion_important']],
                              political_map[sample['political_affilation']],
                              are_arenot_map[sample['is_parent']], do_donot_map_2[sample['uses_media_social']],
                              do_donot_map_2[sample['uses_media_video']],
                              do_donot_map_2[sample['uses_media_news']], do_donot_map_2[sample['uses_media_forums']],
                              tech_impact_map[sample['technology_impact']],
                              toxicpost_map[sample['toxic_comments_problem']]),
            }

            system_prompt = system_prompt_map[prompt_name]
        prompt = partial_prompt % sample['comment']
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
#model_answer = get_annotation(dataset,'gpt-4-1106-preview')

# save all three results
with open(save_directory+dataset_name+prompt_name+'.pkl', 'wb') as handle:
    pickle.dump([model_answer,weighted_answer,scores], handle, protocol=pickle.HIGHEST_PROTOCOL)
