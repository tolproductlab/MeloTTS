import pickle
import os
import re

from . import symbols
from .fr_phonemizer import cleaner as fr_cleaner
from phonemizer import phonemize
from transformers import AutoTokenizer


def distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word
    for task in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word

def text_normalize(text):
    text = fr_cleaner.french_cleaners(text)
    return text

model_id = 'dbmdz/bert-base-french-europeana-cased'
tokenizer = AutoTokenizer.from_pretrained(model_id)

def g2p(text, pad_start_end=True, tokenized=None):
    if tokenized is None:
        tokenized = tokenizer.tokenize(text)
    
    ph_groups = []
    for t in tokenized:
        if not t.startswith("#"):
            ph_groups.append([t])
        else:
            ph_groups[-1].append(t.replace("#", ""))
    
    phones = []
    tones = []
    word2ph = []
    
    for group in ph_groups:
        w = "".join(group)
        phone_len = 0
        word_len = len(group)
        if w == '[UNK]':
            phone_list = ['UNK']
        else:
            # Use phonemizer with espeak backend for French
            # language 'fr-fr' for French, strip trailing whitespace
            phonemes = phonemize(w, language='fr-fr', backend='espeak', strip=True, preserve_punctuation=True)
            phone_list = list(filter(lambda p: p != " ", phonemes))
        
        for ph in phone_list:
            phones.append(ph)
            tones.append(0)
            phone_len += 1
        aaa = distribute_phone(phone_len, word_len)
        word2ph += aaa

    if pad_start_end:
        phones = ["_"] + phones + ["_"]
        tones = [0] + tones + [0]
        word2ph = [1] + word2ph + [1]
    return phones, tones, word2ph

def get_bert_feature(text, word2ph, device=None):
    from text import french_bert
    return french_bert.get_bert_feature(text, word2ph, device=device)

if __name__ == "__main__":
    ori_text = 'Ce service gratuit est“”"" 【disponible》 en chinois 【simplifié] et autres 123'
    text = text_normalize(ori_text)
    print(f"Normalized: {text}")
    phones, tones, word2ph = g2p(text)
    print(f"Phones: {phones}")
