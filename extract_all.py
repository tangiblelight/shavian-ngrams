import re
import sys
from collections import Counter

import pandas as pd
from pathlib import Path
from multiprocessing import Pool
import tqdm

from dechifro.shaw_txt import isolated_lookup


def get_counts(path):
    try:
        df = pd.read_csv(path, sep='\t')
    except:
        print(f'ERROR: {path}', file=sys.stderr)
        return path, {}
    return path, dict(df.itertuples(index=False))


SHAVIAN = '[\U00010450-\U0001047F]'
LATIN = '[a-z]'


def extract_ngram_counts(df, out, alphabet):
    def extract_n_gram(n):
        pattern = re.compile(rf'{alphabet}{{{n}}}')

        def extractor(text):
            return pattern.findall(text.lower())

        return extractor

    def extract_all_n_grams(n, df):
        g = pd.DataFrame({'gram': df['word'].map(extract_n_gram(n)), 'count': df['count']})
        g = g.explode('gram').dropna().groupby('gram')['count'].sum().sort_values(ascending=False)
        return g

    N = 100
    all_ngrams = [
        extract_all_n_grams(1, df)[:N],
        extract_all_n_grams(2, df)[:N],
        extract_all_n_grams(3, df)[:N],
        extract_all_n_grams(4, df)[:N],
    ]
    summary = pd.DataFrame.from_dict({
        '1-grams': all_ngrams[0].index,
        '1-grams count': all_ngrams[0],
        '2-grams': all_ngrams[1].index,
        '2-grams count': all_ngrams[1],
        '3-grams': all_ngrams[2].index,
        '3-grams count': all_ngrams[2],
        '4-grams': all_ngrams[3].index,
        '4-grams count': all_ngrams[3],
    }, orient='index').T

    print(summary)
    summary.to_csv(out)


def main():
    # obtain with https://github.com/pgcorpus/gutenberg get_data.py and process_data.py
    # `data` directory moved to `.cache/gutenberg` to avoid indexing.
    pgcorpus_counts_dir = Path('~/.cache/gutenberg/counts').expanduser()

    output_dir = Path('./counts').expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregated_latin_csv = output_dir.joinpath('aggregate_latin.csv')
    aggregated_shavian_csv = output_dir.joinpath('aggregate_shavian.csv')

    pool = Pool(processes=6)

    if not aggregated_latin_csv.exists():
        # Aggregate counts from all project Gutenberg books. This takes a long time.
        all_counts = Counter()
        glob = list(pgcorpus_counts_dir.glob('*.txt'))
        tqdm_iterator = tqdm.tqdm(enumerate(pool.imap_unordered(get_counts, glob)),
                                  total=len(glob), )
        for i, (path, counts) in tqdm_iterator:
            all_counts.update(counts)

        df = pd.DataFrame(all_counts.items(), columns=['word', 'count'])

        df = df.groupby(['word'])[['count']].sum().sort_values(by='count', ascending=False).reset_index()
        df = df[df['count'] > 5]  # remove very infrequent words

        df.to_csv(aggregated_latin_csv)

    df = pd.read_csv(aggregated_latin_csv, index_col=0)
    extract_ngram_counts(df, output_dir.joinpath('ngrams_latin.csv'), LATIN)

    if not aggregated_shavian_csv.exists():
        print('converting words to shavian...')
        df['word'] = df['word'].map(isolated_lookup)
        print('done.')
        df = df.loc[~df['word'].isna()]
        df = df.groupby('word')['count'].sum().sort_values(ascending=False).reset_index()
        print(df)

        df.to_csv(aggregated_shavian_csv)

    df = pd.read_csv(aggregated_shavian_csv)
    extract_ngram_counts(df, output_dir.joinpath('ngrams_shavian.csv'), SHAVIAN)


if __name__ == '__main__':
    main()
