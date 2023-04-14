#!/usr/bin/env python
# -*- coding: utf-8 -*-
#spellcheck-off
import argparse
import random
import gzip
from tqdm import tqdm
from nltk import word_tokenize

WRITE_BUF_SIZE = 200000

def write_tokenised(set_ids, set_name):
    if args.pretokenise:
        en_file_name = f'{args.output_path}/pretokenised_en_{set_name}.txt'
        fi_file_name = f'{args.output_path}/pretokenised_fi_{set_name}.txt'
    else:
        en_file_name = f'{args.output_path}/raw_en_{set_name}.txt'
        fi_file_name = f'{args.output_path}/raw_fi_{set_name}.txt'
    ids_file_name = f'{args.output_path}/ids_{set_name}.txt'
    with open(en_file_name, 'w', encoding='utf-8') as outen, \
            open(fi_file_name, 'w', encoding='utf-8') as outfi, \
            open(ids_file_name, 'w', encoding='utf-8') as outids:
        buffered_fi = ''
        buffered_en = ''
        buffered_sent_ids = ''
        for i in tqdm(set_ids):
            if args.line2original:
                line_n = id2line[int(i)]
            else:
                line_n = int(i) - 1
            line_en = en_data[line_n]
            line_fi = fi_data[line_n]
            if args.pretokenise:
                line_en = ' '.join(word_tokenize(line_en))
                line_fi = ' '.join(word_tokenize(line_fi))
            if i > 0 and i % WRITE_BUF_SIZE == 0:
                outfi.write(buffered_fi)
                buffered_fi = ''
                outen.write(buffered_en)
                buffered_en = ''
                outids.write(buffered_sent_ids)
                buffered_sent_ids = ''
            buffered_fi += f'{line_fi}\n'
            buffered_en += f'{line_en}\n'
            buffered_sent_ids += f'{i}\n'
        outfi.write(buffered_fi)
        outen.write(buffered_en)
        outids.write(buffered_sent_ids)

def check_filetype_read(filename):
    if filename.endswith('.gz'):
        return gzip.open, 'rt'
    return open, 'r'

def read_file(filename):
    open_func, read_mode = check_filetype_read(filename)
    with open_func(filename, read_mode, encoding='utf-8') as f:
        for line in f.readlines():
            yield line

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pre-tokenise and write data files for ONMT.')
    parser.add_argument('train_set_ids', type=str, help='path to train set ids file')
    parser.add_argument('test_set_ids', type=str, help='path to test set ids file')
    parser.add_argument('fi_data', type=str, help='path to Finnish data')
    parser.add_argument('en_data', type = str, help='path to English data')
    parser.add_argument('output_path', type=str, help='path to output directory')
    parser.add_argument('--line2original', type=str, help='path to line2original file')
    parser.add_argument('--pretokenise', action='store_true', help='Also do pretokenisation.')
    parser.add_argument('--separate-val-data', action='store_true',
        help='Divide the test set into validation and test set.')
    args = parser.parse_args()

    train_set_ids = {int(i.strip()) for i in read_file(args.train_set_ids) if i.strip()}
    test_set_ids = {int(i.strip()) for i in read_file(args.test_set_ids) if i.strip()}
    print('train set size: ', len(train_set_ids))
    print('test set size: ', len(test_set_ids))
    print('intersection size: ', len(train_set_ids.intersection(test_set_ids)))

    if args.line2original:
        id2line = {int(line.strip()): i for i, line in enumerate(read_file(args.line2original))}
    fi_data = [line.strip() for line in read_file(args.fi_data)]
    en_data = [line.strip() for line in read_file(args.en_data)]

    print('fi data size: ', len(fi_data))
    print('en data size: ', len(en_data))

    write_tokenised(train_set_ids, 'train')

    test_set_ids = [int(i) for i in read_file(args.test_set_ids) if i.strip()]

    if args.separate_val_data:
        random.shuffle(test_set_ids)
        print(f'sampling {len(test_set_ids)} test set sentences to get')
        print('validation set with 5000 sentences and test set with 12000 sentences')
        val_set_ids = test_set_ids[:5000]
        test_set_ids_12k = test_set_ids[5000:17000]
        write_tokenised(val_set_ids, 'val')
        write_tokenised(test_set_ids_12k, 'test')
    write_tokenised(test_set_ids, 'test_full')
