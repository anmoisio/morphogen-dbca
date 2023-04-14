#!/usr/bin/env python
# -*- coding: utf-8 -*-
# spellcheck-off
import re
import json
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)


DF_COLUMNS_CONFIDENCE = ['Compound divergence', 'Vocab size', 'seed',
    'BLEU', 'chrF2++',
    'BLEU confidence', 'chrF2++ confidence',
    # 'BLEU confidence mean', 'chrF2++ confidence mean',
    # 'BLEU confidence var', 'chrF2++ confidence var'
    ]


def parse_result_file(f, result_type='confidence'):
    if result_type == 'bleu':
        # the format in test_pred_cp*_full.bleu.chrf2.confidence
        with open(f.name, 'r', encoding='utf-8') as f:
            try:
                json_obj = json.load(f)
            except:
                print(f.name)
        results = {}
        results['BLEU'] = json_obj['score']
        return results
    elif result_type == 'confidence':
        # the format in test_pred_cp*_full.bleu.chrf2.confidence
        with open(f.name, 'r', encoding='utf-8') as f:
            try:
                json_obj = json.load(f)
            except:
                print(f.name)
        results = {}
        for metric in json_obj: # BLEU and chrf2
            results[metric['name']] = {}
            for key in ['score', 'confidence', 'confidence_mean', 'confidence_var']:
                if key in metric:
                    results[metric['name']][key] = metric[key]
                else:
                    results[metric['name']][key] = "None"
        return results
    elif result_type == 'hutmegs':
        with open(f.name, 'r', encoding='utf-8') as f:
            results = {l.split(':')[0]: l.split(':')[1].strip() for l in f.readlines()}
        return results
    elif result_type == 'paired':
        # the format in test_pred_cp*_full.paired-bs
        with open(f.name, 'r', encoding='utf-8') as f:
            json_obj = json.load(f)
        results = {}
        for system in json_obj: # comparing the two NMT models
            results[system['system']] = {}
            for metric in ['BLEU', 'chrF2++']:
                results[system['system']][metric] = system[metric]
        return results


def parse_nmt_filename(filename):
    filename_dict = {}
    filename_dict['Compound divergence'] = float(re.search("comdiv(.+?)[_/]",
                                                           filename).group(1))
    try:
        filename_dict['Vocab size'] = int(re.search("[_/]vocabs[_/](.+?)/",
                                                    filename).group(1).split('_')[0])
    except AttributeError:
        filename_dict['Vocab size'] = int(re.search("[_/]vocabs(.+?)[_/]",
                                                    filename).group(1).split('_')[0])
    filename_dict['seed'] = re.search("_seed(.+?)_", filename)
    if filename_dict['seed']:
        filename_dict['seed'] = filename_dict['seed'].group(1)
    try:
        filename_dict['Tokenisation method'] = re.search(r"_(\w+?)_vocabs",
                                                         filename).group(1)
    except AttributeError:
        pass
    return filename_dict


def parse_tokeniser_filename(filename):
    filename_dict = {}
    filename_dict['Compound divergence'] = float(re.search(r"comdiv(.+?)_",
                                                           filename).group(1))
    filename_dict['Vocab size'] = int(re.search(r"_vocab(\d+?)/",
                                                filename).group(1).split('_')[0])
    filename_dict['seed'] = re.search(r"_seed(.+?)_", filename)
    if filename_dict['seed']:
        filename_dict['seed'] = filename_dict['seed'].group(1)
    filename_dict['Tokenisation method'] = re.search(r"/(\w+?)_vocab",
                                                     filename).group(1)
    return filename_dict


def read_result_files(filenames, result_type='confidence'):
    for filename in filenames:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                results = parse_result_file(f, result_type=result_type)
        except FileNotFoundError:
            print(f"File {filename} not found")
            continue
        if result_type == 'confidence':
            filename_dict = parse_nmt_filename(filename)
            yield filename_dict, results
        elif result_type == 'bleu':
            filename_dict = parse_nmt_filename(filename)
            yield filename_dict, results
        elif result_type == 'hutmegs':
            filename_dict = parse_tokeniser_filename(filename)
            yield filename_dict, results
        elif result_type == 'paired':
            significance_method = filename[-2:]
            for system_name, system_values in results.items():
                filename_dict = parse_nmt_filename(system_name)
                yield filename_dict, significance_method, system_values
        else:
            raise ValueError(f"Unknown result type: {result_type}")


def result_files_to_df(filenames, result_type='confidence'):
    if result_type == 'confidence':
        df = pd.DataFrame(columns=DF_COLUMNS_CONFIDENCE)
        for filename_dict, results in read_result_files(filenames, result_type=result_type):
            df = df.append({**filename_dict, 'BLEU': results['BLEU']['score'],
                'chrF2++': results['chrF2++']['score'],
                'BLEU confidence': results['BLEU']['confidence'],
                'chrF2++ confidence': results['chrF2++']['confidence'],
                'BLEU confidence mean': results['BLEU']['confidence_mean'],
                'chrF2++ confidence mean': results['chrF2++']['confidence_mean'],
                'BLEU confidence var': results['BLEU']['confidence_var'],
                'chrF2++ confidence var': results['chrF2++']['confidence_var']
                },
                ignore_index=True)
        return df
    elif result_type == 'bleu':
        df = pd.DataFrame()
        for filename_dict, results in read_result_files(filenames, result_type=result_type):
            df = df.append({**filename_dict, 'BLEU': results['BLEU'],
                },
                ignore_index=True)
        return df


def significance_files_to_df(filenames):
    df = pd.DataFrame(columns=['Compound divergence', 'Vocab size', 'seed', 'BLEU', 'chrF2++',
        'BLEU_p_value_ar', 'chrF2++_p_value_ar',
        'BLEU_p_value_bs', 'chrF2++_p_value_bs',
        'BLEU_mean_bs', 'chrF2++_mean_bs',
        'BLEU_ci_bs', 'chrF2++_ci_bs'
        ])
    for filename_dict, significance_method, results in \
        read_result_files(filenames, result_type='paired'):
        if results['BLEU']['score'] in df.values:
            # if the system is already in df, only add p_values, means and cis
            df.loc[df['BLEU'] == results['BLEU']['score'],
                f'BLEU_p_value_{significance_method}'] = results['BLEU']['p_value']
            df.loc[df['BLEU'] == results['BLEU']['score'],
                f'chrF2++_p_value_{significance_method}'] = results['chrF2++']['p_value']
            df.loc[df['BLEU'] == results['BLEU']['score'],
                f'BLEU_mean_{significance_method}'] = results['BLEU']['mean']
            df.loc[df['BLEU'] == results['BLEU']['score'],
                f'chrF2++_mean_{significance_method}'] = results['chrF2++']['mean']
            df.loc[df['BLEU'] == results['BLEU']['score'],
                f'BLEU_ci_{significance_method}'] = results['BLEU']['ci']
            df.loc[df['BLEU'] == results['BLEU']['score'],
                f'chrF2++_ci_{significance_method}'] = results['chrF2++']['ci']
        else:
            df = df.append({**filename_dict, 'BLEU': results['BLEU']['score'],
                'chrF2++': results['chrF2++']['score'],
                f'BLEU_p_value_{significance_method}': results['BLEU']['p_value'],
                f'chrF2++_p_value_{significance_method}': results['chrF2++']['p_value'],
                f'BLEU_mean_{significance_method}': results['BLEU']['mean'],
                f'chrF2++_mean_{significance_method}': results['chrF2++']['mean'],
                f'BLEU_ci_{significance_method}': results['BLEU']['ci'],
                f'chrF2++_ci_{significance_method}': results['chrF2++']['ci'],
                },
                ignore_index=True)
    return df


def validation_accs_from_logs(logfiles):
    df = pd.DataFrame(columns=['Step', 'Validation accuracy', 'Compound divergence',
                               'Vocab size', 'seed'])
    for logfile in logfiles:
        with open(logfile, "r") as f:
            lines = f.readlines()
        filenamedict = parse_nmt_filename(logfile)
        for i, line in enumerate(lines):
            if "Validation accuracy" in line:
                accuracy = float(re.search(r"Validation accuracy: (\d+\.\d+)", line).group(1))
                for idx in range(1,5):
                    model_step = re.search(r"Step (\d+)", lines[i + idx])
                    if model_step:
                        model_step = int(model_step.group(1)) - 100
                        break
                    else:
                        model_step = re.search(r"model_step_(\d+)", lines[i + idx])
                        if model_step:
                            model_step = int(model_step.group(1))
                            break
                df = df.append({'Step': model_step, 'Validation accuracy': accuracy,
                    **filenamedict},
                    ignore_index=True)
    return df
