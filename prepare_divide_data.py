#!/usr/bin/env python
#spellcheck-off
"""
This script selects the lemmas and morphological tags that are analysed in the experiment.
The dataset is converted into two matrices that contain the frequencies of atoms and
compounds, respectively, in each sentence. The matrices are saved as pytorch tensors.

The filters are applied in 3 stages. After each stage, files are saved to create a checkpoint.
1. loop through all lemmas (in the conllu file), count frequencies and save lemma types:
    a. remove lemmas that are shorter than 3 characters
    >> save a txt file with all lemma types and their frequencies (ordered by frequency)
2.  2.1 (filter_lemmas())
        b. select only the accepeted lemmas (extracted from a Finnish dictionary)
        c. select lemmas based on frequency (e.g. range 1000-2000 of most frequent lemmas)
    2.2 loop through all sentences and extract the lemmas and their tags (filter_conllu_sents()):
        d. remove tokens with noisy tags, e.g. 'Typo', 'Abbr', 'Foreign'
        e. remove uninteresting morphological tags, e.g. 'Degree=Pos', 'Reflex=Yes'
        f. optionally remove uninteresting combinations of tags i.e. 'Case=Nom+Number=Sing'
    >> save sents_dict, lemma2feats, feat2lemmas, compound_set
3. weight the feats based on how many of the lemmas they appear with (weight_feats()) and
        g. remove compounds with low weights -> filter compound_types
        h. remove lemmas that don't appear with any of the high-weight compounds -> filter lemmas
    >> save
        - ids of atoms (dict), coms (dict) and sents (list)
        - com_weights.pkl
        - used_morph_compounds.txt: list of compounds and their weights
4. create the matrices by looping through the filtered sentences (create_data_matrices()) and
        i. exclude sentences that don't include lemmas after h.
    >> save
        - matrices as pytorch tensors
        - compounds_per_sent_*.txt: list of compounds per sentence
"""
import sys
import argparse
import gzip
import gc
import pickle as pkl
from os import path, makedirs
from collections import Counter
from string import punctuation
from tqdm import tqdm
import torch
from conllu import parse_incr

WRITE_BUF_SIZE = 200000

# lemmas that are not in the Finnish dictionary are removed from the group of analysed words
with open('data/proper-names/all_finnish_lemmas.txt', 'r', encoding='utf-8') as f:
    all_lemmas = [l.strip() for l in f.readlines() if l.strip()]

with open('data/proper-names/english_given_names.txt', 'r', encoding='utf-8') as f:
    english_given_names = [l.strip() for l in f.readlines() if l.strip()]

with open('data/proper-names/finnish_given_names.txt', 'r', encoding='utf-8') as f:
    finnish_given_names = [l.strip() for l in f.readlines() if l.strip()]

with open('data/proper-names/paikannimet_list.txt', 'r', encoding='utf-8') as f:
    places = [l.strip() for l in f.readlines() if l.strip()]

ACCEPTABLE_LEMMAS = set(all_lemmas + english_given_names + finnish_given_names + places)

# words with these POS tags are removed from the group of analysed words
NOISY_POS_TAGS = set() #['PROPN'])

# words with these morphological tags are removed from the group of analysed words
NOISY_TAGS = set(['Typo', 'Abbr', 'Style', 'Foreign', 'NumType', 'AdpType'])

# these tags are removed from the morphological tag list
IGNORED_TAGS = set(['Degree=Pos', 'Reflex=Yes', 'PronType=', 'Derivation='])

# these compounds are ignored when computing compound divergence (atoms are still used)
IGNORED_COMPOUNDS = set() #set(['Case=Nom+Number=Sing'])

# lemmas with these characters are removed from the group of analysed words
EXCLUDED_CHARS = set(punctuation).union(set('1234567890')).union(set(' '))

SEPARATOR = ';'

def check_filetype_read(filename):
    if filename.endswith('.gz'):
        return gzip.open, 'rt'
    return open, 'r'

def read_conllu_file(filename):
    open_func, read_mode = check_filetype_read(filename)
    with open_func(filename, read_mode, encoding='utf-8') as f:
        for tokenlist in tqdm(parse_incr(f)):
            yield tokenlist

def read_lemma(raw_lemma):
    return raw_lemma.strip().strip('#').split('#')[-1].strip().strip('-').split('-')[-1].strip()

def lemma_iter(filename):
    """Iterate over the lemmas in the file."""
    for tokenlist in read_conllu_file(filename):
        for token in tokenlist:
            # filter a. lemmas shorter than 3 characters
            if len(token['lemma']) > 2:
                # if compound word, take only last part of lemma
                yield read_lemma(token['lemma'])

def make_lemma_counter(filename):
    return Counter(lemma_iter(filename)).most_common()

def write_all_lemmas(filename, counter):
    with open(filename, 'w', encoding='utf-8') as f:
        for lemma, freq in counter:
            f.write(f'{lemma} {freq}\n')

def load_lemma_counter(filename):
    open_func, read_mode = check_filetype_read(filename)
    with open_func(filename, read_mode, encoding='utf-8') as f:
        for line in f.readlines():
            split_line = line.strip().split()
            if len(split_line) == 2:
                yield split_line[0].strip(), int(split_line[1])

def lemma_ranges(ranges_str, lemma_counter):
    """The input is in format <start>-<end>-<step>-<nlemmas> or
    <start>-<end>-<step>-auto-<nlemmaoccurrences>
    If the nlemmas is 'auto' take ranges so that the total
    number of word instances stays constant (last int in str) for every range.
    Returns a list of (start, end) tuples for the ranges."""
    lemma_range_list = []
    splitted = [r.split('-') for r in ranges_str.split('.')]
    min_freq = 1
    if len(splitted[-1]) == 1:
        min_freq = int(splitted[-1][0])
        del splitted[-1]
    for range_list in splitted:
        step_size = int(range_list[2])
        steps = range(int(range_list[0]), int(range_list[1]), step_size)
        if range_list[3] == 'auto':
            lemma_occurrences_per_range = int(range_list[4])
            for start in steps:
                occurrences = 0
                nlemmas = 0
                if start >= len(lemma_counter):
                    break
                if lemma_counter[start][1] >= lemma_occurrences_per_range:
                    continue
                if lemma_counter[start][1] < min_freq:
                    break
                for _, freq in lemma_counter[start:]:
                    occurrences += freq
                    nlemmas += 1
                    if occurrences >= lemma_occurrences_per_range:
                        break
                    if nlemmas >= step_size:
                        break
                end = start + nlemmas
                lemma_range_list.append((start, end))
        else:
            n_lemmas = int(range_list[3])
            lemma_range_list += [(start, start + n_lemmas) for start in steps \
                if start < len(lemma_counter) and lemma_counter[start][1] >= min_freq]
    return lemma_range_list

def filter_lemmas(lemma_counts, ranges, output_dir, overwrite=False):
    """First filter out noisy lemmas and then by frequency."""
    print('Using lemma ranges:', ranges)
    most_common = []
    for (lemma, freq) in lemma_counts:
        if len(lemma) > 2 and lemma in ACCEPTABLE_LEMMAS:
            most_common.append((lemma, freq))
    # filter c. lemmas in specific freq ranges
    if ranges == 'all':
        return {lemma for lemma, _ in most_common}
    lemmas = []
    for (start, end) in lemma_ranges(ranges, most_common):
        lemmas_in_range = [lemma for lemma, _ in most_common[start:end]]
        inclusive_end = end - 1
        print(f'Using lemmas in range {start}-{inclusive_end}' + \
            f' with freqs from {most_common[start][1]} to {most_common[inclusive_end][1]}:')
        print('\t' + ', '.join(lemmas_in_range))
        print()
        filename = path.join(output_dir, f'lemma_range_{start}-{inclusive_end}.txt')
        save_struct(lemmas_in_range, filename, overwrite=overwrite)
        lemmas += lemmas_in_range
    lemmas.sort()
    print('\tAfter the first filtering, the lemmas are:', ', '.join(lemmas))
    return set(lemmas)

def filter_token(token, sent_lemmas):
    lemma = read_lemma(token['lemma'])
    if lemma not in sent_lemmas:
        return None
    if token['upos'] in NOISY_POS_TAGS:
        return None
    if not token['feats']:
        return None
    if any(morph_type in NOISY_TAGS for morph_type in token['feats'].keys()):
        return None
    # filter out other rubbish also? semicolon is used as a separator
    if SEPARATOR in token['form']:
        return None
    return lemma

def parse_feats(token):
    feats_string = ''
    for morph_type, morph_class in token['feats'].items():
        morph_tag = f'{morph_type}={morph_class}'
        if any(morph_tag.startswith(badtag) for badtag in IGNORED_TAGS):
            continue
        if feats_string:
            feats_string += '+'
        feats_string += morph_tag
    return feats_string

def filter_conllu_sents(conllu_iterator, used_lemmas, sents_output_file, sent_ids_output_file):
    """Filter the sentences in the conllu file. Write compounds per sent to file.
    Return a dictionary of features to lemmas and the set of compounds."""
    feat2lemmas = {}
    compounds = set()
    buffered_sents = ''
    buffered_sent_ids = ''
    with open(sents_output_file, 'w', encoding='utf-8') as data_out, \
        open(sent_ids_output_file, 'w', encoding='utf-8') as sent_ids_out:
        for i, tokenlist in enumerate(conllu_iterator):
            if i > 0 and i % WRITE_BUF_SIZE == 0:
                print(f'Processed {i} sentences.')
                data_out.write(buffered_sents)
                buffered_sents = ''
                sent_ids_out.write(buffered_sent_ids)
                buffered_sent_ids = ''
            sent_id = tokenlist.metadata['##C: orig_id']
            buffered_sent_ids += sent_id + '\n'
            sent_lemmas = set(read_lemma(token['lemma']) for token in tokenlist
                ).intersection(used_lemmas)
            if not sent_lemmas:
                continue
            sent_line = ''
            for token in tokenlist:
                # parse lemma
                lemma = filter_token(token, sent_lemmas)
                if lemma is None:
                    continue
                # parse morphological tags
                feats = parse_feats(token)
                # create compound
                form_lemma_feats = f'{token["form"]}+{lemma}'
                if feats and feats not in IGNORED_COMPOUNDS:
                    compounds.add(f'{lemma}+{feats}')
                    form_lemma_feats += f'+{feats}'
                    if feats not in feat2lemmas:
                        feat2lemmas[feats] = []
                    feat2lemmas[feats].append(lemma)
                if not sent_line:
                    sent_line = sent_id
                sent_line += SEPARATOR + form_lemma_feats
            if sent_line:
                buffered_sents += f'{sent_line}\n'
        print('Processed all sentences.')
        data_out.write(buffered_sents)
        sent_ids_out.write(buffered_sent_ids)
    return feat2lemmas, compounds

def weight_feats(feat2lemmas):
    """Weight compounds based on the number of different lemmas they appear with.
    >Suppose that the weight of G in this sample is 0.4. Then this means that there exists
    some other subgraph G' that is a supergraph of G in 60% of the occurrences
    of G across the sample set.<"""
    feat_weights_dict = {}
    tot_freqs = {}
    for morph_tag, lemmas in feat2lemmas.items():
        lemma_count = Counter(lemmas)
        total_freq = sum(lemma_count.values())
        tot_freqs[morph_tag] = total_freq
        feat_weights_dict[morph_tag] = 1 - (lemma_count.most_common()[0][1] / total_freq)
    feat_weights_dict = dict(sorted(feat_weights_dict.items(),
        key=lambda item: item[1], reverse=True))
    tot_freqs = dict(sorted(tot_freqs.items(), key=lambda item: item[1], reverse=True))
    return feat_weights_dict, tot_freqs

def final_filter(compounds, feat_weights_dict, comp_w_threshold):
    """Filter compounds with low compound weight. Return atom id dict, compound id dict,
    and the compound weights list (1-D tensor)."""
    com_weights_filtered = {}
    morph_compounds_filtered = set()
    lemmas_filtered = set()
    for com in compounds:
        splitted = str(com.strip()).split('+')
        morph_tag = '+'.join(splitted[1:])
        if morph_tag in feat_weights_dict and feat_weights_dict[morph_tag] >= comp_w_threshold:
            com_weights_filtered[com] = feat_weights_dict[morph_tag]
            morph_compounds_filtered.add(morph_tag)
            lemmas_filtered.add(splitted[0])

    com_filtered = [i for i in compounds if '+'.join(
        i.split('+')[1:]) in morph_compounds_filtered]
    com_filtered.sort()
    compound_ids = {k: i for i, k in enumerate(com_filtered)}
    compound_weights = torch.tensor([com_weights_filtered[i] for i in compound_ids.keys()])

    morph_tags = set()
    for morph in list(morph_compounds_filtered):
        morph_tags.update(morph.split('+'))
    morph_tags = list(morph_tags)
    morph_tags.sort()
    lemmas = list(lemmas_filtered)
    lemmas.sort()
    atoms_ids = {k: i for i, k in enumerate(morph_tags + lemmas)}
    return atoms_ids, compound_ids, compound_weights, morph_compounds_filtered

def make_data_matrices(sent_iter, atomids, comids, filt_morph_coms, coms_per_sent_f):
    atom_dim = len(atomids)
    com_dim = len(comids)
    atom_freq_matrix = torch.zeros((0, atom_dim), dtype=torch.uint8).to_sparse()
    com_freq_matrix = torch.zeros((0, com_dim), dtype=torch.uint8).to_sparse()
    sentence_ids = []
    with open(coms_per_sent_f, 'w', encoding='utf-8') as coms_per_sent_out:
        for sentid, compounds in sent_iter:
            a_row = torch.zeros(atom_dim, dtype=torch.uint8)
            c_row = torch.zeros(com_dim, dtype=torch.uint8)
            writable_compounds = ''
            for compound in compounds:
                splitted = compound.split('+')
                if splitted[1] in atomids: # check lemma in atom_ids
                    for atom in splitted[1:]:
                        if atom in atomids:
                            a_row[atomids[atom]] += 1
                    if '+'.join(splitted[2:]) in filt_morph_coms:
                        c_row[comids['+'.join(splitted[1:])]] += 1
                        writable_compounds += SEPARATOR + compound
            if torch.sum(a_row) > 0:
                atom_freq_matrix = torch.cat((atom_freq_matrix, a_row.unsqueeze(0).to_sparse()), dim=0)
                com_freq_matrix = torch.cat((com_freq_matrix, c_row.unsqueeze(0).to_sparse()), dim=0)
                sentence_ids.append(sentid)
                coms_per_sent_out.write(f'{sentid}{writable_compounds}\n')
    
    return atom_freq_matrix, com_freq_matrix, sentence_ids

def save_struct(struct, filename, overwrite=False):
    if path.isfile(filename) and not overwrite:
        sys.exit(f'{filename} already exists. Use --overwrite or run from a later stage.')
    if filename.endswith('.pkl'):
        with open(filename, 'wb') as pklf:
            pkl.dump(struct, pklf)
    elif filename.endswith('.txt'):
        if isinstance(struct, dict):
            with open(filename, 'w', encoding='utf-8') as txtf:
                for k, v in struct.items():
                    txtf.write(f'{k} {v}\n')
        elif isinstance(struct, (list, set)):
            with open(filename, 'w', encoding='utf-8') as txtf:
                txtf.write('\n'.join(struct) + '\n')
        else:
            sys.exit(f'Cannot save {type(struct)} as a txt file. Supported: list, set, dict.')
    else:
        sys.exit('Unknown file extension. Only .pkl and .txt are supported.')

def load_struct(filename):
    if filename.endswith('.pkl'):
        with open(filename, 'rb') as pklf:
            struct = pkl.load(pklf)
    elif filename.endswith('.txt'):
        with open(filename, 'r', encoding='utf-8') as txtf:
            struct = [c.strip() for c in txtf.readlines()]
    else:
        sys.exit('Unknown file extension. Supported: pkl, txt.')
    return struct

def yield_sents(sents_file):
    with open(sents_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            splitted = line.strip().split(SEPARATOR)
            if len(splitted) > 1:
                yield (splitted[0], splitted[1:])

def main():
    ### Stage 1:
    # count lemma freqs >> save to lemma_freqs_file
    ######
    lemma_freqs_file = path.join(path.dirname(path.normpath(args.output_dir)), 'lemma_freqs.txt')
    if args.stage < 2:
        if path.isfile(lemma_freqs_file) and not args.overwrite:
            raise FileExistsError('Directory already exists. Use --overwrite or run a later stage.')
        makedirs(path.dirname(lemma_freqs_file), exist_ok=True)
        print('\nStage 1:\nCounting lemma frequencies...')
        write_all_lemmas(lemma_freqs_file, make_lemma_counter(args.conllu_file))
    if args.stop_after_stage == 1:
        sys.exit()

    ### Stage 2:
    # filter lemmas, filter sentences
    # >> save sents_dict, lemma2feats, feat2lemmas, compound_types
    ######
    filtered_sents_file = path.join(args.output_dir, 'filtered_sents.txt')
    all_sent_ids_file = path.join(args.output_dir, 'all_sent_ids.txt')
    lemmas_per_feats_file = path.join(args.output_dir, 'lemmas_per_feats.pkl')
    compounds_file = path.join(args.output_dir, 'compounds_stage2.txt')
    if args.stage < 3:
        if path.isdir(args.output_dir) and not args.overwrite:
            raise FileExistsError('Directory already exists. Use --overwrite or run a later stage.')
        makedirs(args.output_dir, exist_ok=True)
        # filter lemmas
        print('\nStage 2:')
        print('Filtering lemmas...')
        filtered_lemmas = filter_lemmas(load_lemma_counter(lemma_freqs_file),
            args.ranges, args.output_dir, overwrite=args.overwrite)
        print('Done filtering lemmas.')
        # filter sentences based on lemmas, and filter noisy compounds
        print('Filtering sentences...')
        lemmas_per_feats, compound_types = filter_conllu_sents(read_conllu_file(args.conllu_file),
            filtered_lemmas, filtered_sents_file, all_sent_ids_file)
        del filtered_lemmas
        gc.collect()
        save_struct(lemmas_per_feats, lemmas_per_feats_file, overwrite=args.overwrite)
        save_struct(compound_types, compounds_file, overwrite=args.overwrite)
    elif args.stage == 3:
        lemmas_per_feats = load_struct(lemmas_per_feats_file)
        compound_types = load_struct(compounds_file)
    if args.stop_after_stage == 2:
        sys.exit()

    ### Stage 3:
    # filter compounds based on weights (given by how many different lemmas they appear with)
    #
    ######
    atom_ids_file = path.join(args.output_dir, 'atom_ids.pkl')
    com_ids_file = path.join(args.output_dir, 'com_ids.pkl')
    com_weights_file = path.join(args.output_dir, 'com_weights.pkl')
    morph_coms_file = path.join(args.output_dir, 'used_morph_compounds.txt')
    if args.stage < 4:
        print('\nStage 3:\nFiltering compounds based on weights...')
        feat_weights, total_freqs = weight_feats(lemmas_per_feats)
        atom_ids, com_ids, com_weights, morph_coms = final_filter(
            compound_types, feat_weights, args.com_weight_threshold)

        output_com_w_file = path.join(args.output_dir, 'morph_com_weights.txt')
        with open(output_com_w_file, 'w', encoding='utf-8') as f:
            f.write('tag\tweight\ttot_freqs\n')
            for tag, weight in feat_weights.items():
                f.write(f'{tag}\t{weight}\t{total_freqs[tag]}\n')
        save_struct(atom_ids, atom_ids_file, overwrite=args.overwrite)
        save_struct(com_ids, com_ids_file, overwrite=args.overwrite)
        save_struct(com_weights, com_weights_file, overwrite=args.overwrite)
        save_struct(morph_coms, morph_coms_file, overwrite=args.overwrite)

        print('Done. Stats:')
        print('\tNumber of atom types:\t', len(atom_ids))
        print('\tNumber of morph feat combinations:\t', len(morph_coms))
        print(f'\tNumber of compound types before: {len(compound_types)},' + \
        f' and after filtering: {len(com_ids)}')

        del feat_weights
        del total_freqs
        del lemmas_per_feats
        del compound_types
        gc.collect()
    elif args.stage == 4:
        atom_ids = load_struct(atom_ids_file)
        com_ids = load_struct(com_ids_file)
        com_weights = load_struct(com_weights_file)
        morph_coms = load_struct(morph_coms_file)
    if args.stop_after_stage == 3:
        sys.exit()

    ### Stage 4:
    # Make the frequency tensors.
    # >> save freq matrices and ids for atoms and coms, save sent_ids
    ######
    if args.stage < 5:
        coms_per_sent_file = path.join(args.output_dir, 'compounds_per_sent.txt')
        atom_m, com_m, sent_ids = make_data_matrices(yield_sents(filtered_sents_file),
            atom_ids, com_ids, morph_coms, coms_per_sent_file)
        if args.weight_compounds:
            # TODO matrix type should be changed to float ?
            raise NotImplementedError('Weighting compounds is not implemented yet.')
            # com_m = torch.multiply(com_m, com_weights)
        save_struct(sent_ids,
            path.join(args.output_dir, 'used_sent_ids.txt'),
            overwrite=args.overwrite)
        atom_freq_file = path.join(args.output_dir, 'atom_freqs.pt')
        com_freq_file = path.join(args.output_dir, 'compound_freqs.pt')
        torch.save(atom_m, atom_freq_file)
        torch.save(com_m, com_freq_file)
        print('Filtering done.')
        print('Data matrix shapes (atoms, compounds):', atom_m.shape, com_m.shape)
        print('Number of sentences used:\t', len(sent_ids))
        # print('Number of compound occurences:\t', int(torch.sum(com_m.to_dense())))
        # print('Number of atom occurences:\t', int(torch.sum(atom_m.to_dense())))

        # finally, save unused sent ids to file
        all_sent_ids = load_struct(all_sent_ids_file)
        all_sent_ids_set = set(all_sent_ids)
        len_all_sent_ids = len(all_sent_ids)
        len_all_sent_ids_set = len(all_sent_ids_set)
        if len_all_sent_ids != len_all_sent_ids_set:
            print('WARNING: duplicate sentence ids in the corpus.')
            print('\tNumber of sentences in the corpus:', len_all_sent_ids)
            print('\tNumber of unique sentence ids in the corpus:', len_all_sent_ids_set)
        unsused_sent_ids = list(all_sent_ids_set - set(sent_ids))
        save_struct(unsused_sent_ids,
            path.join(args.output_dir, 'unused_sent_ids.txt'),
            overwrite=args.overwrite)
        print('Matrices done. Files saved to', args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('conllu_file', type=str, help='Input file in conllu format')
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--stage', type=int, default=0)
    parser.add_argument('--stop_after_stage', type=int, default=-1)
    parser.add_argument('--ranges',  type=str, default='1000-10000-1000-100',
        help='The 4 integers specify ranges of lemma frequencies. <start>-<end>-<step>-<nlemmas>')
    parser.add_argument('--com_weight_threshold', type=float, default=0.5,
        help="Threshold for the weight of a compound to filter out.")
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--weight_compounds', action='store_true')
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()

    if args.profile:
        import cProfile
        cProfile.run('main()')
    else:
        main()
