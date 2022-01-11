#!/usr/bin/env python
# coding: utf-8


import numpy as np
import os
import pandas as pd
import time
import joblib

from sklearn.cluster import KMeans
from ExKMC.Tree import Tree
from find_cut import fit_tree, fit_tree_shallow
from find_cut_makarychev import fit_tree_makarychev

SEEDS = np.arange(1,11)

def get_all_results(jl_list):
    ans = []
    for t in jl_list:
        tree = joblib.load(t)
        res = []
        res.append(get_weighted_depth(tree.tree))
        res.append(get_weighted_depth(tree.tree, False))
        ans.append(res)
    return ans

def get_depths(node, depths, h=0):
    if node is None:
        return
    if node.is_leaf():
        depths.append(h)
    tl = get_depths(node.left, depths, h+1)
    tr = get_depths(node.right, depths, h+1)
    return depths

def get_avg_depth(tree):
    depths = get_depths(tree, [])
    return sum(depths) / len(depths)

def get_cuts_counts(node, cuts, counts, cur_set):
    if (node is None):
        return
    if node.is_leaf():
        cuts.append(len(cur_set))
        counts.append(node.samples)
    else:
        left_set = cur_set.copy()
        left_set.add((node.feature, -1))
        right_set = cur_set.copy()
        right_set.add((node.feature, 1))
        tl = get_cuts_counts(node.left, cuts, counts, left_set)
        tr = get_cuts_counts(node.right, cuts, counts, right_set)
    return cuts, counts

def get_depths_counts(node, depths, counts, h=0):
    if node is None:
        return
    if node.is_leaf():
        depths.append(h)
        counts.append(node.samples)
    tl = get_depths_counts(node.left, depths, counts, h+1)
    tr = get_depths_counts(node.right, depths, counts, h+1)
    return depths, counts

def get_weighted_depth(tree, ignore_repeated=True):
    if ignore_repeated:
        depths, counts = get_cuts_counts(tree, [], [], set())
    else:
        depths, counts = get_depths_counts(tree, [], [])
    n_samples = sum(counts)
    depths = np.array(depths)
    weights = np.array(counts) / n_samples
    return (depths * weights).sum()

def run_exg(data, km, k, max_depth, ratio_type,
            check_imbalance, imbalance_factor,folder=None,
            dataset=None):

    results = {}

    start = time.time()
    tree = Tree(k)
    tree.tree = fit_tree(data, km.cluster_centers_, max_depth, ratio_type,
                         check_imbalance, imbalance_factor)
    y = km.predict(data)
    tree._feature_importance = np.zeros(data.shape[1])
    tree.__fill_stats__(tree.tree, data, y)
    end = time.time()
    # print('folder in run_exg:', folder)
    if folder:
        joblib.dump(tree,f'{folder}/{dataset}_exg_{ratio_type}_seed{km.random_state:02d}.joblib')

    score = tree.score(data)
    max_depth = tree._max_depth()
    avg_depth = get_avg_depth(tree.tree)
    wgtd_depth = get_weighted_depth(tree.tree, False)
    wgtd_cuts = get_weighted_depth(tree.tree)
    secs = end - start

    return score, max_depth, avg_depth, wgtd_depth, wgtd_cuts, secs

def run_shallow(data, km, k, max_depth, ratio_type,
            check_imbalance, imbalance_factor,height_factor,folder=None,
            dataset=None, treat_redundances=False):

    results = {}

    start = time.time()
    tree = Tree(k)
    tree.tree = fit_tree_shallow(data, km.cluster_centers_,height_factor, max_depth, ratio_type,
                         check_imbalance, imbalance_factor, treat_redundances)
    y = km.predict(data)
    tree._feature_importance = np.zeros(data.shape[1])
    tree.__fill_stats__(tree.tree, data, y)
    end = time.time()

    score = tree.score(data)
    max_depth = tree._max_depth()
    avg_depth = get_avg_depth(tree.tree)
    wgtd_depth = get_weighted_depth(tree.tree, False)
    wgtd_cuts = get_weighted_depth(tree.tree)
    secs = end - start
    if folder:
        joblib.dump(tree,f'{folder}/{dataset}_exg_df{height_factor:.2f}_seed{km.random_state:02d}_tr{treat_redundances}.joblib')

    return score, max_depth, avg_depth, wgtd_depth, wgtd_cuts, secs

def test_IMM(data, km, results, k, seed, base_tree='IMM',
             name='IMM', verbose=False, folder=None, dataset=None):
    tree = Tree(k, base_tree=base_tree)
    start = time.time()
    tree = Tree(k, base_tree=base_tree)
    tree.fit(data, km)
    score = tree.score(data)
    end = time.time()
    secs = end - start
    max_depth = tree._max_depth()
    avg_depth = get_avg_depth(tree.tree)
    wgtd_depth = get_weighted_depth(tree.tree, False)
    wgtd_cuts = get_weighted_depth(tree.tree, True)
    if verbose:
        print("{} done in {:.4f} seconds, score = {:.4f}".format(name, secs, score))
        print("Max depth = {}, average depth = {:.2f}, weighted depth = {:.2f}, weighted cuts = {:.2f}".format(max_depth, avg_depth,
                                                                                       wgtd_depth, wgtd_cuts))
        print()
    if folder:
        joblib.dump(tree,f'{folder}/{dataset}_{name}_seed{km.random_state:02d}.joblib')

    results.append({'seed': seed, 'algorithm': name, 'max_depth': max_depth,
                    'avg_depth': avg_depth, 'wgtd_depth': wgtd_depth,
                    'wgtd_cuts': wgtd_cuts, 'score': score, 'time': secs})

def test_greedy(data, km, results, k, seed, max_depth, min_depth, ratio_type,
                check_imbalance, imb_factor, name, verbose=True, folder=None,
                dataset=None):
    while True:
        score, max_depth, avg_depth, wgtd_depth, wgtd_cuts, secs = run_exg(data, km, k,
                                                                max_depth,
                                                                ratio_type,
                                                                check_imbalance,
                                                                imb_factor,
                                                                folder,
                                                                dataset)
        if verbose:

            print("{} done in {:.2f} seconds, score = {:.4f}".format(name, secs, score))
            print("Max depth = {}, average depth = {:.2f}, weighted depth = {:.2f}, weighted cuts = {:.2f}".format(max_depth, avg_depth,
                                                                                           wgtd_depth, wgtd_cuts))
            print()

        res = {'seed': seed, 'algorithm': name,
               'max_depth': max_depth, 'avg_depth': avg_depth,
               'wgtd_depth': wgtd_depth, 'wgtd_cuts': wgtd_cuts,
               'score': score, 'time': secs}
        results.append(res)

        if max_depth <= min_depth:
            break
        else:
            max_depth -= 1

def test_shallow(data, km, results, k, seed, max_depth, min_depth, ratio_type,
                check_imbalance, imb_factor, name, height_factor, verbose=True,
                folder=None, dataset=None, treat_redundances=False):

    score, max_depth, avg_depth, wgtd_depth, wgtd_cuts, secs = run_shallow(data, km, k,
                                                            max_depth,
                                                            ratio_type,
                                                            check_imbalance,
                                                            imb_factor,height_factor,
                                                            folder=folder,
                                                            dataset=dataset,
                                                            treat_redundances=treat_redundances)
    if verbose:

        print("{} done in {:.2f} seconds, score = {:.4f}".format(name, secs, score))
        print("Max depth = {}, average depth = {:.2f}, weighted depth = {:.2f}, weighted cuts = {:.2f}".format(max_depth, avg_depth,
                                                                                       wgtd_depth, wgtd_cuts))
        print()

    res = {'seed': seed, 'algorithm': name,
           'max_depth': max_depth, 'avg_depth': avg_depth,
           'wgtd_depth': wgtd_depth, 'wgtd_cuts': wgtd_cuts,
           'score': score, 'time': secs}
    results.append(res)

def test_makarychev(data, km, results, k, seed, dataset, folder=None,
                    verbose=True, name='makarychev'):

    start = time.time()
    tree = Tree(k)
    tree.tree = fit_tree_makarychev(data, km.cluster_centers_)
    y = km.predict(data)
    tree._feature_importance = np.zeros(data.shape[1])
    tree.__fill_stats__(tree.tree, data, y)
    score = tree.score(data)
    end = time.time()
    secs = end - start
    max_depth = tree._max_depth()
    avg_depth = get_avg_depth(tree.tree)
    wgtd_depth = get_weighted_depth(tree.tree, False)
    wgtd_cuts = get_weighted_depth(tree.tree, True)
    if verbose:
        print("{} done in {:.4f} seconds, score = {:.4f}".format(name, secs, score))
        print("Max depth = {}, average depth = {:.2f}, weighted depth = {:.2f}, weighted cuts = {:.2f}".format(max_depth, avg_depth,
                                                                                       wgtd_depth, wgtd_cuts))
        print()
    if folder:
        joblib.dump(tree,f'{folder}/{dataset}_{name}_seed{km.random_state:02d}.joblib')

    results.append({'seed': seed, 'algorithm': 'makarychev', 'max_depth': max_depth,
                    'avg_depth': avg_depth, 'wgtd_depth': wgtd_depth,
                    'wgtd_cuts': wgtd_cuts, 'score': score, 'time': secs})

def test_depths(data, k, algos, seed=None, verbose=True, folder=None,
                dataset=None):
    results = []
    max_depth = k-1
    min_depth = int(np.ceil(np.log2(k)))

    print("for seed", seed)

    # kmeans
    start = time.time()
    km = KMeans(k, random_state=seed)
    km.fit(data)
    score = -km.score(data)
    secs = time.time() - start
    print("kmeans done in {:.4f} seconds, score = {:.4f}".format(secs, score))
    print()
    res = {'seed': seed, 'algorithm': 'kmeans', 'max_depth': None,
           'avg_depth': None, 'wgtd_depth': None, 'wgtd_cuts': None,
           'score': score, 'time': secs}
    results.append(res)

    # IMM
    if 'IMM' in algos:
        test_IMM(data, km, results, k, seed, 'IMM', 'IMM', verbose)

    # Ex-KMC
    if 'Ex-KMC' in algos:
        test_IMM(data, km, results, k, seed, 'NONE', 'KMC', verbose)

    # Ex-Greedy (with loose ratio)
    if 'Ex-Greedy' in algos:
        test_greedy(data, km, results, k, seed, max_depth, min_depth,
                    "loose", False, 1, 'Ex-Greedy (loose)', verbose,
                    folder, dataset)


    return pd.DataFrame(results)


def test_depths_seeds(data, k, algos, seeds, name=None, folder=None):
    df = test_depths(data, k, algos, seeds[0], folder, name)
    file_name = name + "_results_temp.csv"
    df.to_csv(file_name, index=False)
    for seed in seeds[1:]:
        new_df = test_depths(data, k, algos, seed)
        df = pd.concat([df, new_df])
        df.to_csv(file_name, index=False)

    if name:
        df['dataset'] = name

    kmeans_vals = df[(df.algorithm == 'kmeans')][['seed', 'score']]
    kmeans_vals.index = kmeans_vals.seed
    full_km_vals = kmeans_vals.loc[df.seed].score.values
    df['PoE'] = df.score / full_km_vals

    os.remove(file_name)

    return df

def test_and_save(data, k, algos, seeds, name, folder):
    """
    data: numpy array (data to cluster)
    k: integer (number of clusters)
    seeds: list (list of seeds for the k-means algo)
    name: string (name of the file)
    folder: string (folder in which to save the file)
    """
    df = test_depths_seeds(data, k, algos, seeds, name=name, folder=folder)
    #retrieve results for ExGreedy
    # gb = df[df.algorithm=='Ex-Greedy (loose)'].groupby(['dataset','seed']).max_depth.max()

    # check_depths = df.apply(lambda x: gb[x.dataset][x.seed],axis=1) == df.max_depth
    # ex_greedys = df[(df.algorithm=='Ex-Greedy (loose)') &
    #                 (check_depths)].copy()
    # ex_greedys.algorithm = 'Ex-Greedy'
    # df = pd.concat([df, ex_greedys])
    # print(df)
    csv_name = name + "_results.csv"
    df.to_csv(folder + csv_name, index=False)
