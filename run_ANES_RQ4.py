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

base_dir='./dataset/ANES/Results_RQ4/'
save_directory=base_dir+save_name+'/'

if not os.path.exists(save_directory):
    os.makedirs(save_directory)


dataset=pd.read_csv('./dataset/ANES/anes2012_full_sampled.csv')

with open('./dataset/ANES/prompt_'+dataset_name+'.txt', 'r') as f:
    partial_prompt = f.read()
partial_prompt=partial_prompt[:-14]
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

if dataset_name=='relig_pray':
    choice_to_score = {'A': 1, 'B': 2,'C': 3, 'D': 4, 'E': 5}
elif dataset_name=='presapp_econ':
    choice_to_score = {'A': 1, 'B': 2}
elif dataset_name=='ecblame_pres':
    choice_to_score = {'A': 1, 'B': 2,'C': 3, 'D': 4, 'E': 5}
elif dataset_name=='presapp_foreign':
    choice_to_score = {'A': 1, 'B': 2}
elif dataset_name=='ecblame_fmpr':
    choice_to_score = {'A': 1, 'B': 2,'C': 3, 'D': 4, 'E': 5}
elif dataset_name=='spsrvpr_ssself':
    choice_to_score = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F':6, 'G':7}
elif dataset_name=='govrole_big':
    choice_to_score = {'A': 1, 'B': 2}
elif dataset_name=='ecblame_dem':
    choice_to_score = {'A': 1, 'B': 2,'C': 3, 'D': 4, 'E': 5}
elif dataset_name=='ptywom_bettrpty':
    choice_to_score = {'A': 1, 'B': 2,'C': 3}
elif dataset_name=='resent_deserve':
    choice_to_score = {'A': 1, 'B': 2,'C': 3, 'D': 4, 'E': 5}
elif dataset_name=='ident_amerid':
    choice_to_score = {'A': 1, 'B': 2,'C': 3, 'D': 4, 'E': 5}
elif dataset_name == 'aidblack_self':
    choice_to_score = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5,'F':6, 'G':7}
elif dataset_name == 'egal_toofar':
    choice_to_score = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
elif dataset_name == 'trad_famval':
    choice_to_score = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
elif dataset_name=='gayrt_marry':
    choice_to_score = {'A': 1, 'B': 2,'C': 3}
elif dataset_name == 'prmedia_attvnews':
    choice_to_score = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
elif dataset_name=='nonmain_bias':
    choice_to_score = {'A': 1, 'B': 2,'C': 3}
elif dataset_name=='interest_following':
    choice_to_score = {'A': 1, 'B': 2,'C': 3}
elif dataset_name=='gayrt_adopt':
    choice_to_score = {'A': 1, 'B': 2}
elif dataset_name == 'effic_undstd':
    choice_to_score = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
elif dataset_name == 'immig_checks':
    choice_to_score = {'A': 1, 'B': 2, 'C': 3}


def prepare_data(sample, prompt_name):
    race_map= {1: 'white', 2: 'black', 3: 'asian', 4: 'native American', 5: 'hispanic', 6: 'other non-hispanic including multiple races'}
    ideology_map= {1: "extremely liberal", 2: "liberal",
               3: "slightly liberal", 4: "moderate",
               5: "slightly conservative", 6: "conservative",
               7: "extremely conservative"}
    discuss_politics_map= {1: 'I like to discuss politics with my family and friends.',
               2: 'I never discuss politics with my family or friends.'}
    party_id_map={1: "a strong democrat", 2: "a weak Democrat",
     3: "an independent who leans Democratic", 4: "an independent",
     5: "an independent who leans Republican", 6: "a weak Republican",
     7: "a strong Republican"}
    gender_map= {1: "man", 2: "woman"}
    flag_map={1:"extremely good", 2:"very good", 3:"moderately good", 4:"slightly good", 5:"not good at all"}
    church_map={1:"attend church", 2:"do not attend church"}
    state_map = {
        1: "Alabama",
        2: "Alaska",
        4: "Arizona",
        5: "Arkansas",
        6: "California",
        8: "Colorado",
        9: "Connecticut",
        10: "Delaware",
        11:"Washington, D.C.",
        12: "Florida",
        13: "Georgia",
        15: "Hawaii",
        16: "Idaho",
        17: "Illinois",
        18: "Indiana",
        19: "Iowa",
        20: "Kansas",
        21: "Kentucky",
        22: "Louisiana",
        23: "Maine",
        24: "Maryland",
        25: "Massachusetts",
        26: "Michigan",
        27: "Minnesota",
        28: "Mississippi",
        29: "Missouri",
        30: "Montana",
        31: "Nebraska",
        32: "Nevada",
        33: "New Hampshire",
        34: "New Jersey",
        35: "New Mexico",
        36: "New York",
        37: "North Carolina",
        38: "North Dakota",
        39: "Ohio",
        40: "Oklahoma",
        41: "Oregon",
        42: "Pennsylvania",
        44: "Rhode Island",
        45: "South Carolina",
        46: "South Dakota",
        47: "Tennessee",
        48: "Texas",
        49: "Utah",
        50: "Vermont",
        51: "Virginia",
        53: "Washington",
        54: "West Virginia",
        55: "Wisconsin",
        56: "Wyoming"}
    interest_map={1:"very", 2:"somewhat", 3:"slightly", 4:"not at all"}

    instruction_prompt = partial_prompt

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
        system_prompt_map= {
        "Default": """**It is 2012. Your Profile**: Racially, you are %s. You are a %s. You are %s years old. Ideologically, you are %s. Politically, you are %s. It makes you feel %s when you see the American flag flying. You %s. You are %s interested in politics and public affairs.""" % (
        race_map[int(sample['race'])], gender_map[int(sample['gender'])], int(sample['age']),
        ideology_map[int(sample['ideology'])], party_id_map[int(sample['party_id'])], flag_map[int(sample['flag'])], church_map[int(sample['attend_church'])],interest_map[int(sample['interest'])]),
        "Fewer_SV":"""**It is 2012. Your Profile**: Racially, you are %s. You are a %s. You are %s years old. It makes you feel %s when you see the American flag flying. You %s. You are %s interested in politics and public affairs.""" % (
        race_map[int(sample['race'])], gender_map[int(sample['gender'])], int(sample['age']),
        flag_map[int(sample['flag'])], church_map[int(sample['attend_church'])],interest_map[int(sample['interest'])])

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

    desired_tokens = list(choice_to_score.keys())
    scores = list(choice_to_score.values())
    desired_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in desired_tokens]
    final_answers=[]
    weighted_final_answers=[]
    probabilities = []

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


