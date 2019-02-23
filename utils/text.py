# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: text.py

@time: 2019/2/1 14:01

@desc:

"""

import re
from nltk.stem.porter import PorterStemmer


def get_tokens_from_parse(parse):
    """Parse a string in the binary tree SNLI format and return a string of joined by space tokens"""
    cleaned = parse \
        .replace('(', ' ').replace(')', ' ') \
        .replace('-LRB-', '(').replace('-RRB-', ')') \
        .replace('-LSB-', '[').replace('-RSB-', ']')

    tokens = cleaned.split()

    cleaned_string = ' '.join(tokens)

    # remove all non-ASCII characters for MetaMap
    cleaned_string = cleaned_string.encode('ascii', errors='ignore').decode()

    return cleaned_string


def is_time(token):
    without_time = re.sub(r'(\d)*(\d):(\d\d)([aA][mM]|[pP][Mm])', '', token).strip()
    return not without_time


def is_num(token):
    try:
        num = float(token)
        return True
    except ValueError as e:
        return False


def is_non_alpha(token):
    without_alpha = re.sub(r'[^A-Za-z]', '', token).strip()
    return not without_alpha


def remove_punctuation(token):
    without_punctuation = re.sub(r'[^A-Za-z0-9]', '', token)
    return without_punctuation


def clean_unit(text):
    text = re.sub(r"(\d+)kgs ", lambda m: m.group(1) + ' kg ', text)  # e.g. 4kgs => 4 kg
    text = re.sub(r"(\d+)kg ", lambda m: m.group(1) + ' kg ', text)  # e.g. 4kg => 4 kg
    text = re.sub(r"(\d+)k ", lambda m: m.group(1) + '000 ', text)  # e.g. 4k => 4000
    text = re.sub(r"\$(\d+)", lambda m: m.group(1) + ' dollar ', text)
    text = re.sub(r"(\d+)\$", lambda m: m.group(1) + ' dollar ', text)
    return text


def remove_acronym(text):
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"cannot", "can not ", text)
    text = re.sub(r"what\'s", "what is", text)
    text = re.sub(r"What\'s", "what is", text)
    text = re.sub(r"\'ve ", " have ", text)
    text = re.sub(r"n\'t", " not ", text)
    text = re.sub(r"i\'m", "i am ", text)
    text = re.sub(r"I\'m", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"c\+\+", "cplusplus", text)
    text = re.sub(r"c \+\+", "cplusplus", text)
    text = re.sub(r"c \+ \+", "cplusplus", text)
    text = re.sub(r"c#", "csharp", text)
    text = re.sub(r"f#", "fsharp", text)
    text = re.sub(r"g#", "gsharp", text)
    text = re.sub(r" e mail ", " email ", text)
    text = re.sub(r" e \- mail ", " email ", text)
    text = re.sub(r" e\-mail ", " email ", text)
    text = re.sub(r",000", '000', text)
    text = re.sub(r"\'s", " ", text)
    return text


def spell_correct(text):
    text = re.sub(r"ph\.d", "phd", text)
    text = re.sub(r"PhD", "phd", text)
    text = re.sub(r"pokemons", "pokemon", text)
    text = re.sub(r"pokémon", "pokemon", text)
    text = re.sub(r"pokemon go ", "pokemon-go ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" 9 11 ", " 911 ", text)
    text = re.sub(r" j k ", " jk ", text)
    text = re.sub(r" fb ", " facebook ", text)
    text = re.sub(r"facebooks", " facebook ", text)
    text = re.sub(r"facebooking", " facebook ", text)
    text = re.sub(r"insidefacebook", "inside facebook", text)
    text = re.sub(r"donald trump", "trump", text)
    text = re.sub(r"the big bang", "big-bang", text)
    text = re.sub(r"the european union", "eu", text)
    text = re.sub(r" usa ", " america ", text)
    text = re.sub(r" us ", " america ", text)
    text = re.sub(r" u s ", " america ", text)
    text = re.sub(r" U\.S\. ", " america ", text)
    text = re.sub(r" US ", " america ", text)
    text = re.sub(r" American ", " america ", text)
    text = re.sub(r" America ", " america ", text)
    text = re.sub(r" quaro ", " quora ", text)
    text = re.sub(r" mbp ", " macbook-pro ", text)
    text = re.sub(r" mac ", " macbook ", text)
    text = re.sub(r"macbook pro", "macbook-pro", text)
    text = re.sub(r"macbook-pros", "macbook-pro", text)
    text = re.sub(r" 1 ", " one ", text)
    text = re.sub(r" 2 ", " two ", text)
    text = re.sub(r" 3 ", " three ", text)
    text = re.sub(r" 4 ", " four ", text)
    text = re.sub(r" 5 ", " five ", text)
    text = re.sub(r" 6 ", " six ", text)
    text = re.sub(r" 7 ", " seven ", text)
    text = re.sub(r" 8 ", " eight ", text)
    text = re.sub(r" 9 ", " nine ", text)
    text = re.sub(r"googling", " google ", text)
    text = re.sub(r"googled", " google ", text)
    text = re.sub(r"googleable", " google ", text)
    text = re.sub(r"googles", " google ", text)
    text = re.sub(r" rs(\d+)", lambda m: ' rs ' + m.group(1), text)
    text = re.sub(r"(\d+)rs", lambda m: ' rs ' + m.group(1), text)
    text = re.sub(r"the european union", " eu ", text)
    text = re.sub(r"dollars", " dollar ", text)
    return text


def clean_sentence(sentence):
    sentence = clean_unit(sentence)
    sentence = remove_acronym(sentence)
    sentence = spell_correct(sentence)

    tokens = sentence.split(' ')
    sentence_cleaned_tokens = []
    for i, token in enumerate(tokens):
        if is_time(token):
            sentence_cleaned_tokens.append('TIME')
            continue

        if is_num(token):
            sentence_cleaned_tokens.append('NUM')
            continue

        if is_non_alpha(token):
            continue

        token = remove_punctuation(token)
        sentence_cleaned_tokens.append(token)

    sentence_cleaned = ' '.join(sentence_cleaned_tokens)
    return sentence_cleaned


def clean_list_of_sentences(sentences):
    sentences_cleaned = [clean_sentence(s) for s in sentences]
    return sentences_cleaned


def clean_data(data):
    if data is None:
        return None

    premise_cleaned = clean_list_of_sentences(data['premise'])
    hypothesis_cleaned = clean_list_of_sentences(data['hypothesis'])

    data['premise'] = premise_cleaned
    data['hypothesis'] = hypothesis_cleaned

    return data


def stem_list_of_sentences(sentences, stemmer):
    sentences = [s.split() for s in sentences]
    sentences = [[stemmer.stem(t) for t in s] for s in sentences]
    sentences = [' '.join(s) for s in sentences]

    return sentences


def stem_data(data):
    if data is None:
        return None

    stemmer = PorterStemmer()

    premise_stemmed = stem_list_of_sentences(data['premise'], stemmer)
    hypothesis_stemmed = stem_list_of_sentences(data['hypothesis'], stemmer)

    data['premise'] = premise_stemmed
    data['hypothesis'] = hypothesis_stemmed

    return data


def lowecase_list_of_sentences(sentences):
    sentences_lowercased = [s.lower() for s in sentences]
    return sentences_lowercased


def lowercase_data(data):
    if data is None:
        return None

    premise_lowercased = lowecase_list_of_sentences(data['premise'])
    hypothesis_lowercased = lowecase_list_of_sentences(data['hypothesis'])

    data['premise'] = premise_lowercased
    data['hypothesis'] = hypothesis_lowercased

    return data
