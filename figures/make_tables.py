#!/usr/bin/env python
# -*- coding: utf-8 -*-
#spellcheck-off


def significances():
    """ Tables 2 and 4 in the Nodalida2023 paper """
    df = significance_files_to_df(args.result_files).round(decimals=2)
    # print(df)

    df['BLEU'] = df['BLEU'].astype(str) + ' (' + df['BLEU_mean_bs'].astype(str) + \
        ' ± ' + df['BLEU_ci_bs'].astype(str) + ')'
    df['chrF2++'] = df['chrF2++'].astype(str) + ' (' + df['chrF2++_mean_bs'].astype(str) + \
        ' ± ' + df['chrF2++_ci_bs'].astype(str) + ')'

    df['Vocab size'] = df['Vocab size'].astype(int)

    df['BLEU_p_value_bs'].fillna(value=np.nan, inplace=True)
    df['BLEU_p_value_bs'] = 'p = ' +  df['BLEU_p_value_bs'].round(decimals=4).astype(str)
    df['chrF2++_p_value_bs'].fillna(value=np.nan, inplace=True)
    df['chrF2++_p_value_bs'] = 'p = ' +  df['chrF2++_p_value_bs'].round(decimals=4).astype(str)

    df = df.drop(['BLEU_mean_bs', 'BLEU_ci_bs', 'chrF2++_mean_bs', 'chrF2++_ci_bs'], axis=1)

    df = df.pivot(index=['seed', 'Vocab size'],
                  columns='Compound divergence',
                  values=['chrF2++', 'BLEU', 'chrF2++_p_value_bs', 'BLEU_p_value_bs']
                  )

    # print(df)
    print(df.to_latex())


def all_vocabs_with_confidence():
    """ Table 3 in the Nodalida2023 paper """
    df = result_files_to_df(args.result_files, result_type='confidence').round(decimals=2)
    # print(pivot_df(df))

    # sort df by vocab size and compound divergence
    df = df.sort_values(by=['Vocab size', 'Compound divergence'])

    df['BLEU'] = df['BLEU'].astype(str) + ' (' + df['BLEU confidence mean'].astype(str) + \
        ' ± ' + df['BLEU confidence var'].astype(str) + ')'
    df['chrF2++'] = df['chrF2++'].astype(str) + \
        ' (' + df['chrF2++ confidence mean'].astype(str) + \
        ' ± ' + df['chrF2++ confidence var'].astype(str) + ')'
    df = df.drop(['BLEU confidence', 'chrF2++ confidence'], axis=1)

    df_chrf = df.pivot(index=['Vocab size'],
                    columns='Compound divergence',
                    values=['chrF2++'])
    print(df_chrf.to_latex())
    df_bleu = df.pivot(index=['Vocab size'],
                    columns='Compound divergence',
                    values=['BLEU'])
    print(df_bleu.to_latex())


if __name__ == '__main__':
    import argparse
    from df_utils import *
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_files',  nargs='*', type=str)
    parser.add_argument('--type', type=str, default='significances')
    args = parser.parse_args()

    if args.type == 'significances':
        significances()
    elif args.type == 'all_vocabs_with_confidence':
        all_vocabs_with_confidence()
    else:
        raise ValueError(f'Unknown type: {args.type}')
