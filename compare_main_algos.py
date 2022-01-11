import numpy as np
import pandas as pd
import time
import os
from sklearn.cluster import KMeans

from get_results import test_shallow, test_IMM, test_greedy
from get_datasets import *

DATASETS = ['anuran', 'avila', 'beer', 'bng', 'cifar10', 'collins', 'covtype',
            'digits', 'iris', 'letter', 'mice', 'newsgroups', 'pendigits',
            'poker', 'sensorless', 'vowel']

SEEDS = np.arange(1,31)

FOLDER = 'results'

VERBOSE = True

DEPTH = 0.03


def compare_main_algos(datasets, seeds, depth, folder, verbose):
    for dataset in datasets:
        full_df = pd.DataFrame(columns=['seed', 'algorithm', 'max_depth', 'avg_depth',
                                   'wgtd_depth', 'score', 'time', 'dataset', 'PoE',
                                   'depth_factor'])
        print("for", dataset)
        data, k = eval('get_{}()'.format(dataset))
        print("shape:", data.shape, "k:", k)
        for s in seeds:
            print('for seed', s)
            results = []

            # kmeans
            start = time.time()
            km = KMeans(k, random_state=s)
            km.fit(data)
            score = -km.score(data)
            secs = time.time() - start
            print("kmeans done in {:.4f} seconds, score = {:.4f}".format(secs, score))
            print()
            res = {'seed': s, 'algorithm': 'kmeans', 'max_depth': None,
                   'avg_depth': None, 'wgtd_depth': None, 'score': score,
                   'time': secs}
            results.append(res)
            # joblib.dump(km,f'{folder}/{dataset}_km_seed{km.random_state:02d}.joblib')

            # ExShallow
            test_shallow(data=data, km=km, results=results, k=k, seed=s,
                         max_depth=k, min_depth=1,ratio_type="none",
                         check_imbalance=False,imb_factor=1,name="ExShallow",
                         height_factor=depth, verbose=verbose, dataset=dataset,
                         treat_redundances=True)
            results[-1]['depth_factor'] = depth

            df = pd.DataFrame(results)
            df['dataset'] = dataset

            kmeans_vals = df[(df.algorithm == 'kmeans')][['seed', 'score']]
            kmeans_vals.index = kmeans_vals.seed
            full_km_vals = kmeans_vals.loc[df.seed].score.values
            df['PoE'] = df.score / full_km_vals
            full_df = pd.concat([full_df, df])

            # IMM
            test_IMM(data=data, km=km, results=results, k=k, seed=s,
                     verbose=verbose, dataset=dataset)
            df = pd.DataFrame(results)
            df['dataset'] = dataset

            kmeans_vals = df[(df.algorithm == 'kmeans')][['seed', 'score']]
            kmeans_vals.index = kmeans_vals.seed
            full_km_vals = kmeans_vals.loc[df.seed].score.values
            df['PoE'] = df.score / full_km_vals
            full_df = pd.concat([full_df, df])

            # ExKMC
            test_IMM(data=data, km=km, results=results, k=k, seed=s,
                     base_tree='NONE', name='KMC', verbose=verbose,
                     dataset=dataset)

            # ExGreedy
            test_greedy(data=data, km=km, results=results, k=k, seed=s,
                        max_depth=k-1, min_depth=k-1, ratio_type="loose",
                        check_imbalance=False, imb_factor=1, name='Ex-Greedy',
                        verbose=verbose, dataset=dataset)

            df = pd.DataFrame(results)
            df['dataset'] = dataset

            kmeans_vals = df[(df.algorithm == 'kmeans')][['seed', 'score']]
            kmeans_vals.index = kmeans_vals.seed
            full_km_vals = kmeans_vals.loc[df.seed].score.values
            df['PoE'] = df.score / full_km_vals
            full_df = pd.concat([full_df, df])

        print(full_df.shape)
        if FOLDER not in os.listdir():
            os.mkdir(FOLDER)
        full_df.drop_duplicates().to_csv(f'./{folder}/{dataset}_results.csv')

if __name__ == '__main__':
    compare_main_algos(DATASETS, SEEDS, DEPTH, FOLDER, VERBOSE)
