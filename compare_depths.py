import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans

from get_results import test_shallow, test_IMM, test_greedy
from get_datasets import *

DATASETS = ['anuran', 'avila', 'beer', 'bng', 'cifar10', 'collins', 'covtype',
            'digits', 'iris', 'letter', 'mice', 'newsgroups', 'pendigits',
            'poker', 'sensorless', 'vowel']

SEEDS = [1] #np.arange(1,11)

FOLDER = 'test'

VERBOSE = True

DEPTHS = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.15,0.20,0.25,0.30,
          0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00]

for dataset in DATASETS:
    full_df = pd.DataFrame(columns=['seed', 'algorithm', 'max_depth', 'avg_depth',
                               'wgtd_depth', 'score', 'time', 'dataset', 'PoE',
                               'depth_factor'])
    print("for", dataset)
    data, k = eval('get_{}()'.format(dataset))
    print("shape:", data.shape, "k:", k)
    for s in SEEDS:
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
        for d in DEPTHS:
            test_shallow(data=data, km=km, results=results, k=k, seed=s,
                         max_depth=k, min_depth=1,ratio_type="none",
                         check_imbalance=False,imb_factor=1,name="ExShallow",
                         height_factor=d,verbose=VERBOSE, dataset=dataset,
                         treat_redundances=True)
            results[-1]['depth_factor'] = d

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
    full_df.drop_duplicates().to_csv(f'./{FOLDER}/{dataset}_results_depth.csv')
