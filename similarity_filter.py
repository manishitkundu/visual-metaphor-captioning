import pickle
na_map = pickle.load(open('Cleaned_NA_Map','rb'))
an_map = pickle.load(open('Cleaned_AN_Map','rb'))

import json
with open('/workspaces/manishit_project/Controlled Metaphor Generation/datasets/text/ChainNet/data/chainnet_simple/chainnet_metaphor.json', 'r') as file:
    data = json.load(file)['content']

print(data)

chainnet_metaphor_nouns = []
for i in data:
    chainnet_metaphor_nouns.append(i['wordform'])
chainnet_metaphor_nouns = list(set(chainnet_metaphor_nouns))

print(len(chainnet_metaphor_nouns))

from tqdm import tqdm
from nltk.corpus import wordnet as wn


## Template based generation

def generator(PAS):
    Primary, Attribute, Secondary = PAS
    generations = []
    if Secondary[0].lower() in 'aeiou':
        generations.append(f"The {Primary} is as {Attribute} as an {Secondary}.")
        generations.append(f"His {Primary} is as {Attribute} as an {Secondary}")
        generations.append(f"Her {Primary} is as {Attribute} as an {Secondary}")
    else:
        generations.append(f"The {Primary} is as {Attribute} as a {Secondary}.")
        generations.append(f"His {Primary} is as {Attribute} as a {Secondary}")
        generations.append(f"Her {Primary} is as {Attribute} as a {Secondary}")

    return generations
        
generated_similes = []
n_keys = na_map.keys()
a_keys = an_map.keys()
for primary in tqdm(chainnet_metaphor_nouns):
    if primary in n_keys:
        for attribute in na_map[primary]:
            if attribute in a_keys:
                for secondary in an_map[attribute]:
                    # sentence = f"The {primary} is as {attribute} as a {secondary}."
                    for sentence in generator((primary,attribute,secondary)):
                        generated_similes.append([sentence, (primary,attribute,secondary)])

print(f"Number of generated sentences is {len(generated_similes)}")

## Filtering based on similarity

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def sbert_similarity_filtering(word,list_of_words):
    sentences = [word] + list_of_words
    embeddings = model.encode(sentences)
    scores = (embeddings[0]@embeddings[1:].T)
    reduced_list_of_words = []
    for i in range(len(scores)):
        if scores[i] < 0.5:
            reduced_list_of_words.append(list_of_words[i])
    
    return list(zip(reduced_list_of_words,scores))

def rule_based_similarty_nfiltering(word,list_of_words):
    reduced_list_of_words = []
    for candidate in list_of_words:
        if not ((candidate in word) or (candidate[:-1] in word)):
            if not ((word in candidate) or (word[:-1] in candidate)):
                reduced_list_of_words.append(candidate)

    return reduced_list_of_words

def rule_based_similarty_afiltering(word,list_of_words):
    reduced_list_of_words = []
    for candidate in list_of_words:
        if not ((word in candidate) or (word[:-1] in candidate)):
            if not ((candidate in word) or (candidate[:-1] in word)):
                reduced_list_of_words.append(candidate)

    return reduced_list_of_words

delete_items = []
for item_no in range(len(generated_similes)):
    item = generated_similes[item_no]
    sentence = item[0]
    primary, attribute, secondary = item[1]

    if not rule_based_similarty_nfiltering(primary,[secondary]):
        delete_items.append(item)
    elif not rule_based_similarty_nfiltering(attribute,[secondary]):
        delete_items.append(item)
    elif not rule_based_similarty_afiltering(primary,[attribute]):
        delete_items.append(item)

print(f"Length before deletion: ",len(generated_similes))
for item in tqdm(delete_items):
    generated_similes.remove(item)
print(f"Length after deletion: ",len(generated_similes))


delete_items = []
for item_no in tqdm(range(len(generated_similes))):
    item = generated_similes[item_no]
    sentence = item[0]
    primary, attribute, secondary = item[1]

    if not sbert_similarity_filtering(primary, [secondary]):
        delete_items.append(item)

print(f"Length before deletion: ",len(generated_similes))
for item in tqdm(delete_items):
    generated_similes.remove(item)
print(f"Length after deletion: ",len(generated_similes))

with open('filtered_similes','wb') as f:
    pickle.dump(generated_similes,f)