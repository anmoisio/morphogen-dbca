#!/usr/bin/env python
"""
Funtions for detokenising tokenised sentences and desubwording
subworded sentences.
"""
# -*- coding: utf-8 -*-
from nltk import word_tokenize

def desubword(sent) -> list:
    """Convert a sentence (list or string) of subwords into a list of whole words."""
    if isinstance(sent, list):
        sent = ' '.join(sent)
    return sent.strip().replace(' ', '').replace('▁', ' ').split()

def make_subword2word(subword_sent: list) -> dict:
    """Returns a dictionary that maps a subword index to a word index."""
    if not subword_sent[0].startswith('▁'):
        return {k: k for k in range(len(subword_sent))}
    subword2word = {}
    word_idx = -1
    for i, token in enumerate(subword_sent):
        if token.startswith('▁'):
            word_idx += 1
        subword2word[i] = word_idx
    return subword2word

def make_detokenise(original_sent, tokenised_sent) -> dict:
    """Returns a dictionary that maps a tokenised index to a word index."""
    detokenise = {}
    tokenised_idx = 0
    subwords_combined = ''
    for orig_idx, word in enumerate(original_sent):
        while word != subwords_combined:
            detokenise[tokenised_idx] = orig_idx
            if tokenised_idx >= len(tokenised_sent):
                print(f'ERROR: {word} is longer than {subwords_combined} in {original_sent}')
                return detokenise
            subwords_combined += tokenised_sent[tokenised_idx]
            tokenised_idx += 1
            if len(subwords_combined) > len(word):
                subwords_combined = subwords_combined.replace("''", '"').replace("``", '"').replace('``', '"')
                if len(subwords_combined) > len(word):
                    print(f'ERROR: {subwords_combined} is longer than {word} in {original_sent}')
                    return detokenise
        subwords_combined = ''
    return detokenise

def detokenise_alignment(src_sent, tgt_sent, alig):
    """Converts a tokenised alignment into a raw text alignment."""
    if isinstance(src_sent, str): src_sent = src_sent.split()
    if isinstance(tgt_sent, str): tgt_sent = tgt_sent.split()
    if isinstance(alig, str): alig = alig.split()
    src_sw2w = make_detokenise(src_sent, word_tokenize(' '.join(src_sent)))
    tgt_sw2w = make_detokenise(tgt_sent, word_tokenize(' '.join(tgt_sent)))
    try:
        detokenised = [(src_sw2w[pair[0]], tgt_sw2w[pair[1]]) for pair in alig]
    except KeyError:
        print(f'Error in detokenising alignment.')
        return []
    return detokenised

def desubword_alignment(src_sent, tgt_sent, alig):
    """Converts a subword alignment into a word alignment."""
    if isinstance(src_sent, str): src_sent = src_sent.split()
    if isinstance(tgt_sent, str): tgt_sent = tgt_sent.split()
    if isinstance(alig, str): alig = alig.split()
    src_sw2w = make_subword2word(src_sent)
    tgt_sw2w = make_subword2word(tgt_sent)
    return [(src_sw2w[i[0]], tgt_sw2w[i[1]]) for i in alig]
