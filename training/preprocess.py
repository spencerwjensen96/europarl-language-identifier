import os
import re
import math

import pandas as pd
import numpy as np

root_path = "/Users/spencer.jensen/Desktop/code/language_detector"
data_path = "/Users/spencer.jensen/Desktop/code/language_detector/training/data/txt"
csv_path = "/Users/spencer.jensen/Desktop/code/language_detector/training/data/csv"


def combine_texts():
    languages = [f for f in os.listdir(data_path) if not f.endswith('.txt')]
    for lang in languages:
        if os.path.exists(os.path.join(data_path, lang, f'{lang}.txt')):
            continue

        files = [os.path.join(data_path, lang, f) for f in os.listdir(os.path.join(data_path, lang))]

        lines = []
        for file in files:
            with open(file, 'rb') as f:
                s = f.read()
            lines.append(s)
        with open(os.path.join(data_path, lang, f'{lang}.txt'), 'wb') as f:
            for line in lines:
                f.write(line)
        print(f'completed {lang}')


def process_into_csv_df():
    corpora = [c for c in os.listdir(data_path) if c.endswith('.txt') and not c.endswith('cleaned.txt')]

    for corpus in corpora:
        if os.path.exists(os.path.join(csv_path, corpus.replace('.txt', '.csv'))):
            continue
        lang = corpus.replace('.txt', '')
        print('{} .. cleaning and converting to a csv .. '.format(corpus), end='\n')
        in_text = open(os.path.join(data_path, corpus), 'rb').read().decode('utf-8',errors='ignore')
        cleaned = re.sub(r'<.*?>', '', str(in_text)).lower().strip()
        cleaned = re.sub(r'\(([^\)]{2})\)', '', cleaned)
        cleaned = re.sub(r'([0-9]+[., -]*)', '', cleaned)
        outfile = corpus.replace('.txt', '-cleaned.txt')
        open(os.path.join(data_path, outfile), 'w').write(cleaned)

        df = pd.read_table(os.path.join(data_path, outfile), on_bad_lines='skip', header=None, names=['text'])
        df['lang'] = lang
        df = df[['lang', 'text']]
        df.drop_duplicates()
        df.to_csv(os.path.join(csv_path, corpus.replace('.txt', '.csv')), index=False, header=False)
    print('finished.')

def remove_small_text():
    corpora = [c for c in os.listdir(csv_path)]
    for corpus in corpora:
        lang = corpus[:2]
        df = pd.read_csv(os.path.join(csv_path, corpus), names=['lang', 'text'])
        indexes = []
        [ indexes.append(i) for i, j in enumerate(df['text']) if len(j) < 15]
        df = df.drop(indexes)
        df = df[['lang', 'text']]
        df.to_csv(os.path.join(csv_path, f'{lang}-cleaned.csv'), index=False, header=False)
        print(f"processed {lang} successfully")


def normalize_text(row):
    label = '__label__' + str(row['lang'])
    txt = str(row['text'])

    return ' '.join((label + ' , ' + txt).split())

def split_train_test():
    df = pd.read_csv(os.path.join(root_path, 'training/data/europarl.csv'), names=['lang', 'text'])
    df = df.reindex(np.random.permutation(df.index)).reset_index(drop=True)
    df['normalized'] = df.apply(lambda row: normalize_text(row), axis=1)
    split = 80000
    train = df['normalized'][:split].copy()
    test = df['normalized'][split:].copy()

    np.savetxt(os.path.join(root_path, 'training/fastText/europarl.train'), train.values, fmt="%s")
    np.savetxt(os.path.join(root_path, 'training/fastText/europarl.eval'), test.values, fmt="%s")

#combine_texts()

#process_into_csv_df()

#remove_small_text()

split_train_test()
