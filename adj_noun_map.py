import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet', quiet=True)
from numpy import exp
import numpy as np
import pickle
from tqdm import tqdm

adic = pickle.load(open('adj_dictionary','rb'))
ndic = pickle.load(open('noun_dictionary','rb'))
nouns = list(ndic.keys())
adjs = list(adic.keys())
for i in 'QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm':
    if i in nouns:
        nouns.remove(i)

import gensim.downloader as api

word_emb_model2 = api.load('glove-wiki-gigaword-300')


## Word vector based mapping

def get_associated_words(word, topn=100, model_no = 1):

    if model_no == 1:
        model = word_emb_model1
    elif model_no == 2:
        model = word_emb_model2
    elif model_no == 3:
        model = word_emb_model3
    else:
        print("No such model.")
        return None

    try:
        # Get the most similar words
        similar_words = model.most_similar(word, topn=topn)
        # Extract only the words from the result
        associated_words = [word for word, similarity in similar_words]
        return associated_words
    except KeyError:
        return f"The word '{word}' is not in the vocabulary of the model."

def synonyms(word):
    synos = []
    for syn in wn.synsets(word):
        for lem in syn.lemmas():
            synos.append(lem.name())
    return synos

def wordnet_based_secondary(word):
    secondaries = []
    for syno in synonyms(word):
        if 'like' in syno.lower():
            concept = syno.replace('like','')
            if concept in nouns:
                secondaries.append(concept)
    return secondaries

def Secondary_Generator(word, top_k = 100):
    synonym = synonyms(word)
    associated_words = get_associated_words(word,1000,2)
    s_list = []
    for associated_word in associated_words:
        if associated_word in nouns:
            if associated_word not in synonym:
                s_list.append(associated_word)

    return s_list[:top_k]


## Mapping using Roberta Masked Predictions


import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForMaskedLM.from_pretrained('roberta-base')

def get_top_predictions(masked_sentence, top_k=5):
    
    inputs = tokenizer(masked_sentence, return_tensors='pt')
    mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    mask_token_logits = logits[0, mask_token_index, :]
    top_tokens = torch.topk(mask_token_logits, top_k, dim=1).indices[0].tolist()

    top_words = [tokenizer.decode([token]).strip() for token in top_tokens]
    return top_words

def Bert_Secondary_Generator(word):
    combined_top = []
    for masked_sentence in [f"as {word} as <mask>", f"as {word} as the <mask>", f"as {word} as a <mask>", f"as {word} as an <mask>"]:
        top_predictions = get_top_predictions(masked_sentence, top_k=20)
        combined_top += top_predictions
        # print(f"Top predictions for '{masked_sentence}': {top_predictions}")
    combined_top = list(set(combined_top))
    return combined_top
    

## Pooling generations from different mappings

def pooled_secondary_generator(word):
    bert_pred = Bert_Secondary_Generator(word)
    wv_pred = Secondary_Generator(word)
    wn_pred = wordnet_based_secondary(word)

    pooled_generation = []
    for i in wv_pred:
        if i in bert_pred:
            pooled_generation.append(i)
    
    pooled_generation += wn_pred
    pooled_generation = list(set(pooled_generation))

    return pooled_generation


print("Mapping begins...")

an_map = {}
iteration_no = 0
for adj in tqdm(adjs):
    if adj not in an_map.keys():
        an_map[adj] = pooled_secondary_generator(adj)

    iteration_no += 1


    if iteration_no%100 == 0:
        pickle.dump(an_map, open('AN_Map','wb'))


print("Saving!")
with open('AN_Map','wb') as f:
    pickle.dump(an_map,f)