import nltk
from nltk.corpus import wordnet as wn
import evaluate
import transformers
nltk.download('wordnet', quiet=True)
from numpy import exp
import numpy as np
import pickle

adic = pickle.load(open('adj_dictionary','rb'))
ndic = pickle.load(open('noun_dictionary','rb'))
nfull = pickle.load(open('noun_full_dictionary','rb'))

print("Length of adjective dictionary: ",len(adic))
print("Length of full noun dictioanry: ",len(nfull))
print("Lenght of filtered noun dictionary: ",len(ndic))

nouns = list(nfull.keys())
adjs = list(adic.keys())

# def properties_of_word(word):

#     syn = wn.synset(word)
#     attributes = [func for func in dir(syn) if func[0] != "_"]

#     for attribute in attributes:
#         # Get the attribute
#         attr = getattr(syn, attribute)
#         # print()
#         # Check if the attribute is callable (i.e., it is a method)
#         if callable(attr):
#             try:
#                 # Call the method
#                 if attr():
#                     print(f"{attribute} : {attr()}")
#             except TypeError as e:
#                 # print(f"Could not call {attribute}(): {e}")
#                 # print()
#                 continue

# #properties_of_word("march.n.01")


## Wordnet based extraction

def adj_extract_wn(word):
    syn = wn.synset(word)
    add = []

    if syn.hyponyms():
        for item in syn.hyponyms():
            if '.n.' in item.name():
                hypo_name = item.name().split('.n.')[0]
                if "_"+syn.name().split(".n.")[0] in hypo_name:
                    add.append(hypo_name.replace("_"+syn.name().split(".n.")[0],""))


    syn_name = syn.name().split('.n.')[0]
    if syn.examples():
        for item in syn.examples():
            if syn_name in item:
                split_sent = item.split(" ")
                for token in range(len(split_sent)):
                    if split_sent[token] == syn_name:
                        if split_sent[token - 1] in adjs:
                            add.append(split_sent[token - 1])
                            break
    return add


def all_synset_adj_extractor(word):
    adj_map = []
    for syn in wn.synsets(word):
        if word+".n." in syn.name():
            adj_map += adj_extract_wn(syn.name())

    for adj in adj_map:
        if "'" in adj:
            adj_map.remove(adj)
    return(adj_map)


## Llama-based generation

import transformers
import torch

print("Loading LLaMa 3.1")

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16,
    "quantization_config": {"load_in_4bit": True}},
    device_map="cuda",
    token = <api-key>,
)

print("Loading done.")


def generate_5_adjs_llm(word, definition):
    messages = [
        {"role": "system", "content": "You are an English expert who always answers accurately and concisely."},
        {"role": "user", "content": f"Here is a word, '{word}', meaning '{definition}'. Return the top 3 associated adjectives with this word. If the definition does not match the word well, then return the top 3 most associated adjectives with the definition. Only list the words, do not add additional remarks or comments at the beginning or at the end. It is important that you return just the adjectives."},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=50,
        pad_token_id = 128001
    )
    result = outputs[0]["generated_text"][-1]['content']
    output = []
    for item in result.split("\n"):
        output.append(item)
    return output

def all_synsets_llm_gen(word):
    adj_map = []
    for syn in wn.synsets(word):
        if word+".n." in syn.name():
            syn_name = syn.name().split(".n.")[0]
            definition = syn.definition()
            # print(syn_name,definition)
            adj_map += generate_5_adjs_llm(syn_name,definition)

    adj_map = list(set(adj_map))

    for adj in adj_map:
        if adj.lower() not in adjs:
            adj_map.remove(adj)

    for i in range(len(adj_map)):
        adj_map[i] = adj_map[i].lower()
    return(adj_map)


print("Wordnet Extraction Started!")

from tqdm import tqdm
noun_adj_dictionary = {}
for noun in tqdm(nouns):
    if noun not in noun_adj_dictionary.keys():
        noun_adj_dictionary[noun] = all_synset_adj_extractor(noun)


print("Wordnet Extraction Done!")
print("LLM Generation Started!")

count = 0
for noun in tqdm(noun_adj_dictionary.keys()):
    noun_adj_dictionary[noun] += all_synsets_llm_gen(noun)
    noun_adj_dictionary[noun] = list(set(noun_adj_dictionary[noun]))

    count += 1

    if count%100 == 0:
        with open('Noun_Adj_Complete_Map','wb') as f:
            pickle.dump(noun_adj_dictionary,f)




print("LLM Generation Done!")
print("Saving...")



with open('Noun_Adj_Complete_Map','wb') as f:
    pickle.dump(noun_adj_dictionary,f)


print("Saved!")