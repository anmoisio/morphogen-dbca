#!/usr/bin/env python
import unittest
import torch
from prepare_divide_data import *
# e.g.
# python -m unittest tests.test_prepare_divide_data.TestPrepareDivideData.test_weight_feats

class TestPrepareDivideData(unittest.TestCase):
    """Unittests for the prepare_divide_data module."""

    def test_make_lemma_counter(self):
        self.assertEqual(make_lemma_counter('tests/data/example_parsed2.txt')[:4],
            [('olla', 96), ('tämä', 20), ('tai', 19), ('joka', 19)])

    def test_write_all_lemmas(self):
        input_file = 'tests/data/example_parsed2.txt'
        output_file = 'tests/data/example_lemmas2.txt'
        write_all_lemmas(output_file, make_lemma_counter(input_file))
        with open(output_file, 'r', encoding='utf-8') as f:
            self.assertTrue(f.read().startswith('olla 96\ntämä 20\ntai 19\njoka 19\n'))

    def test_filter_lemmas(self):
        lemma_counter = [('olla', 96), ('tämä', 50),
                        ('Toshiba', 44), ('EMU', 33),
                        ('tai', 19), ('joka', 16),
                        ('vai', 15), ('2ta', 10),
                        ('hän', 9), ('t.h', 8),
                        ('auto', 7), ('pyörä', 6),
                        ('tööt', 9)]
        self.assertEqual(filter_lemmas(lemma_counter, ranges='all', output_dir='tests/data'),
            set(['olla', 'tämä', 'tööt', 'tai', 'joka', 'vai', 'hän', 'auto', 'pyörä']))

    def test_filter_conllu_sents(self, print_sents=False):
        filtered_lemmas = set(['äyriäinen', 'tämä', 'se', 'jalkainen'])
        sent_iter = read_conllu_file('tests/data/example_parsed.txt')
        filtered_sents_file = 'tests/data/example_filtered_sents.txt'
        filtered_sent_ids_file = 'tests/data/example_filtered_sents.txt'
        feat2lemmas, compounds = filter_conllu_sents(sent_iter, filtered_lemmas,
            filtered_sents_file, filtered_sent_ids_file)

        sorted_atoms = []
        for sentid, sentence in yield_sents(filtered_sents_file):
            sorted_atoms_sent = []
            for word in sentence:
                print(word)
                sorted_atoms_sent += word.split('+')[1:]
            sorted_atoms += [sorted(sorted_atoms_sent)]

        sorted_compounds = []
        for sentid, sentence in yield_sents(filtered_sents_file):
            sorted_compounds.append(sorted(['+'.join(w.split('+')[1:]) for w in sentence]))

        sorted_form_lemma_feats = [sorted(sentence) for _, sentence
            in yield_sents(filtered_sents_file)]

        # self.assertEqual(int(filtered_sents[0][0]), 85951678)
        self.assertEqual(sorted_atoms[0], sorted(['tämä', 'Case=Gen', 'Case=Ine',
            'Clitic=Kin', 'Number=Sing', 'Number=Sing', 'se']))
        self.assertEqual(sorted_atoms[1], sorted(['äyriäinen', 'se',
            'jalkainen', 'Case=Nom', 'Number=Plur',
            'Case=Ela', 'Case=Nom', 'Number=Plur', 'Number=Sing']))

        self.assertEqual(sorted_compounds[0], sorted([
            'se+Case=Gen+Clitic=Kin+Number=Sing',
            'tämä+Case=Ine+Number=Sing']))
        self.assertEqual(sorted_compounds[1], sorted(['se+Case=Ela+Number=Sing',
            'äyriäinen+Case=Nom+Number=Plur',
            'jalkainen+Case=Nom+Number=Plur',]))

        self.assertEqual(sorted_form_lemma_feats[0], sorted([
            'senkin+se+Case=Gen+Clitic=Kin+Number=Sing',
            'tässä+tämä+Case=Ine+Number=Sing']))
        self.assertEqual(sorted_form_lemma_feats[1], sorted([
            'siitä+se+Case=Ela+Number=Sing',
            'äyriäiset+äyriäinen+Case=Nom+Number=Plur',
            'pääjalkaiset+jalkainen+Case=Nom+Number=Plur']))
        if print_sents:
            for sentid, sent in yield_sents(filtered_sents_file):
                print(sentid)
                for d in sent[1]:
                    print(d)
        self.assertEqual(feat2lemmas.keys(), set(['Case=Gen+Clitic=Kin+Number=Sing',
            'Case=Ine+Number=Sing', 'Case=Ela+Number=Sing', 'Case=Nom+Number=Plur']))
        self.assertEqual(sorted(compounds), sorted(['tämä+Case=Ine+Number=Sing',
            'se+Case=Gen+Clitic=Kin+Number=Sing',
            'se+Case=Ela+Number=Sing',
            'äyriäinen+Case=Nom+Number=Plur',
            'jalkainen+Case=Nom+Number=Plur']))

    def test_weight_feats(self):
        lemmas_per_feat = {'Case=Ill+Number=Sing': ['voima', 'mikä'],
            'Case=Ill+Number=Plur': ['asu', 'asu'],
            'Case=Ela+Number=Sing': ['asu', 'asu', 'mikä'],
            }
        feat_weights_dict, tot_freqs = weight_feats(lemmas_per_feat)
        self.assertEqual(feat_weights_dict['Case=Ill+Number=Sing'], 0.5)
        self.assertEqual(feat_weights_dict['Case=Ill+Number=Plur'], 0.0)
        self.assertAlmostEqual(feat_weights_dict['Case=Ela+Number=Sing'], 1/3)
        self.assertEqual(tot_freqs['Case=Ill+Number=Sing'], 2)
        self.assertEqual(tot_freqs['Case=Ill+Number=Plur'], 2)
        self.assertEqual(tot_freqs['Case=Ela+Number=Sing'], 3)

    def test_final_filter(self):
        coms = ['tämä+Case=Ill+Number=Sing', 'auto+Case=Ill+Number=Sing',
            'tapaus+Case=Ine+Number=Sing', 'se+Case=Ela+Number=Sing',
            'asu+Case=Ela+Number=Sing']
        feat_weights = {'Case=Ill+Number=Sing': 0.5,
            'Case=Ine+Number=Sing': 0.0,
            'Case=Ela+Number=Sing': 1/3,
            'Case=Gen+Clitic=Kin+Number=Sing': 0.0,
            'Case=Nom+Number=Plur': 0.0,}
        atom_ids, com_ids, com_weights, morph_compounds_filtered = final_filter(coms,
            feat_weights, 0.2)
        self.assertEqual(atom_ids, {
            'Case=Ela': 0, 'Case=Ill': 1, 'Number=Sing': 2,
            'asu': 3, 'auto': 4, 'se': 5, 'tämä': 6})
        self.assertEqual(com_ids, {
            'asu+Case=Ela+Number=Sing': 0, 'auto+Case=Ill+Number=Sing': 1,
            'se+Case=Ela+Number=Sing': 2, 'tämä+Case=Ill+Number=Sing': 3})
        for a, b in zip(list(com_weights), [1/3, 0.5, 1/3, 0.5]):
            self.assertAlmostEqual(a, b)
        self.assertEqual(morph_compounds_filtered, set(['Case=Ill+Number=Sing',
            'Case=Ela+Number=Sing']))

    def test_make_data_matrices(self):
        sents_dict = {
            123: ['autoon+auto+Case=Ill+Number=Sing',
                    'siihen+se+Case=Ill+Number=Sing'], # filt out, 'se' is not in lemmas
            666: ['sen+se+Case=Gen+Number=Sing', # whole sent filtered out, no lemmas
                'siihen+se+Case=Ill+Number=Sing'],
            456: ['talolle+talo+Case=All+Number=Sing',
                'siitä+se+Case=Ela+Number=Sing'], # filt out, 'se' is not in lemmas
            789: ['talolle+talo+Case=All+Number=Sing',
                'autolle+auto+Case=All+Number=Sing',
                'autossa+auto+Case=Ine+Number=Sing',], # filt out, Case=Ine+Number=Sing is not in filtered_morph_coms
            34: ['siitä+se+Case=Ela+Number=Sing', # coms filtered out, 'auto' lemma stays
                'autossa+auto+Case=Ine+Number=Sing',],
            }
        atom_ids = {'auto': 0, 'Case=Ill': 1, 'Number=Sing': 2, 'Case=Ela': 3,
            'talo': 4, 'Case=All': 5}
        com_ids = {'auto+Case=Ill+Number=Sing': 0, 'auto+Case=All+Number=Sing': 1,
            'talo+Case=All+Number=Sing': 2}
        filtered_morph_coms = set(['Case=All+Number=Sing', 'Case=Ill+Number=Sing'])
        coms_per_sent_file = 'tests/data/coms_per_sent.txt'
        atom_freq_matrix, com_freq_matrix, sent_ids = make_data_matrices(
            sents_dict.items(), atom_ids, com_ids, filtered_morph_coms, coms_per_sent_file)
        torch.equal(atom_freq_matrix.to_dense(), torch.tensor(
            [[1, 1, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 1],
            [1, 0, 2, 0, 1, 2],
            [1, 0, 0, 0, 0, 0],], dtype=torch.uint8))
        torch.equal(com_freq_matrix.to_dense(), torch.tensor(
            [[1, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [0, 0, 0]], dtype=torch.uint8))
        self.assertEqual(sent_ids, [123, 456, 789, 34])
        with open(coms_per_sent_file, 'r') as f:
            lines = f.readlines()
        self.assertEqual(lines,
            ['123;autoon+auto+Case=Ill+Number=Sing\n',
            '456;talolle+talo+Case=All+Number=Sing\n',
            '789;talolle+talo+Case=All+Number=Sing;autolle+auto+Case=All+Number=Sing\n',
            '34\n'
            ])
    
    def test_filter_token(self):
        token = {}
        token['lemma'] = 'auto'
        token['feats'] = {'Case': 'Ill', 'Number': 'Sing'}
        token['form'] = 'autoon'
        token['upos'] = 'NOUN'
        self.assertEqual(filter_token(token, set(['auto'])), 'auto')

        token['lemma'] = 'pyörä'
        token['feats'] = {'Case': 'Ill', 'Number': 'Sing'}
        self.assertEqual(filter_token(token, set(['auto'])), None)

        token['lemma'] = 'auto'
        token['feats'] = {'Case': 'Ill', 'Number': 'Sing', 'Typo': 'Yes'}
        self.assertEqual(filter_token(token, set(['auto'])), None)

        token['lemma'] = 'auto'
        token['form'] = 'polku;auto'
        token['feats'] = {'Case': 'Ill', 'Number': 'Sing'}
        self.assertEqual(filter_token(token, set(['auto'])), None)

    def test_parse_feats(self):
        token = {}
        token['feats'] = {'Case': 'Ill', 'Number': 'Sing'}
        self.assertEqual(parse_feats(token), 'Case=Ill+Number=Sing')
        token['feats'] = {'Degree': 'Pos', 'Number': 'Sing'}
        self.assertEqual(parse_feats(token), 'Number=Sing')
        token['feats'] = {'Case': 'Ill', 'Derivation': 'Inen', 'Number': 'Sing'}
        self.assertEqual(parse_feats(token), 'Case=Ill+Number=Sing')


if __name__ == '__main__':
    unittest.main()
