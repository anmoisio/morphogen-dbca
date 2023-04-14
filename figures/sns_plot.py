#!/usr/bin/env python
# -*- coding: utf-8 -*-

def opus_test():
    """ Figure 1 in the Nodalida paper. """
    df = result_files_to_df(args.result_files, result_type='confidence')
    print(df)
    # exit()
    sns.set_theme(style="white")
    relplot = sns.relplot(
        kind="line",
        data=df,
        x='Compound divergence',
        y="BLEU",
        palette='blues',
    )
    relplot.set(xlabel='Training set compound divergence')
    relplot.set(ylim=(43, 45.7))
    relplot.fig.set_size_inches(4, 3)
    relplot.fig.savefig(args.output, dpi=400, bbox_inches = 'tight')

def all_vocabs():
    """ Figure 2 in the Nodalida paper. """
    df = result_files_to_df(args.result_files, result_type='confidence')
    sns.set_theme(style="white")
    relplot = sns.relplot(
        kind="line",
        data=df,
        x='Compound divergence',
        y="BLEU",
        hue='Vocab size',
        palette='rocket',
    )
    relplot.set(xlabel='Compound divergence')
    sns.move_legend(relplot, "upper left", bbox_to_anchor=(0.5, 1.0), ncol=1)
    relplot.fig.set_size_inches(4, 5)
    relplot.fig.savefig(args.output, dpi=400)

def subplot_seeds():
    """ Figure 3 in the Nodalida paper. """
    per_seed_per_vocab = result_files_to_df(args.result_files, result_type='confidence')
    seeds = [[11,22],[33,44],[55,66],[77,88]] # 4 rows of 2
    # seeds = [[11,22,33,44],[55,66,77,88]] # 2 rows of 4
    n_rows = len(seeds)
    n_cols = len(seeds[0])
    sns.set_theme(style="white")
    fig, axes = plt.subplots(n_rows, n_cols)
    for row in range(n_rows):
        for col in range(n_cols):
            seed = seeds[row][col]
            df = per_seed_per_vocab[per_seed_per_vocab["seed"] == str(seed)]
            relplot = sns.lineplot(
                data=df,
                x='Compound divergence',
                # y="BLEU",
                y="chrF2++",
                hue='Vocab size',
                palette='deep',
                ax=axes[row][col],
            )
            relplot.set(xlabel='Compound divergence')
    # fig.set_size_inches(23, 11) # 2 rows of 4
    fig.set_size_inches(11, 17) # 4 rows of 2
    fig.savefig(args.output, dpi=400, bbox_inches = 'tight')

if __name__ == '__main__':
    import argparse
    from df_utils import result_files_to_df
    import seaborn as sns
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_files',  nargs='*', type=str, required=True)
    parser.add_argument('--output', type=str, default='figures/untitled.png')
    parser.add_argument('--type', type=str, required=True)
    args = parser.parse_args()

    if args.type == 'opus_test':
        opus_test()
    elif args.type == 'all_vocabs':
        all_vocabs()
    elif args.type == 'subplot_seeds':
        subplot_seeds()
    else:
        raise ValueError('Unknown type: ' + args.type)
