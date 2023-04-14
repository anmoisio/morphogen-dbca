#!/usr/bin/env python
import nltk
import argparse

parser = argparse.ArgumentParser(
    description='Tokenize a file with one sentence per line using nltk.word_tokenize.')
parser.add_argument('corpus_path', type=str, help='path to corpus')
parser.add_argument('output_path', type=str, help='path to output file')
args = parser.parse_args()

with open(args.corpus_path, 'r', encoding='utf-8') as f, \
    open(args.output_path, 'w', encoding='utf-8') as fout:
    for line in f:
        fout.write(' '.join(nltk.word_tokenize(line.strip())) + '\n')
