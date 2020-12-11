#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""

from Base.NonPersonalizedRecommender import TopPop, Random
from EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Hybrid.HybridCombinationSearch import HybridCombinationSearch, HybridCombinationMergedSearch
from Hybrid.HybridCombinationSearchCV import HybridCombinationSearchCV
from Hybrid.LinearHybrid002ggg import LinearHybrid002ggg
from Hybrid.LinearHybrid002 import LinearHybrid002
from Hybrid.LinearHybrid001 import LinearHybrid001
from Hybrid.MergedHybrid000 import MergedHybrid000
from Hybrid.PipeHybrid001 import PipeHybrid001
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNN_CBF_CF import ItemKNN_CBF_CF
from MatrixFactorization.IALSRecommender import IALSRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, \
    MatrixFactorization_FunkSVD_Cython
from MatrixFactorization.PureSVDRecommender import PureSVDRecommender, PureSVDItemRecommender
from MatrixFactorization.NMFRecommender import NMFRecommender

import traceback

import os, multiprocessing
from functools import partial
from itertools import combinations

from DataParser import DataParser
from Data_manager.Movielens.Movielens10MReader import Movielens10MReader
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

from ParameterTuningCV.run_parameter_search_CV import runParameterSearch_Collaborative


def read_data_split_and_search():
    """
    This function provides a simple example on how to tune parameters of a given algorithm

    The BayesianSearch object will save:
        - A .txt file with all the cases explored and the recommendation quality
        - A _best_model file which contains the trained model and can be loaded with recommender.load_model()
        - A _best_parameter file which contains a dictionary with all the fit parameters, it can be passed to recommender.fit(**_best_parameter)
        - A _best_result_validation file which contains a dictionary with the results of the best solution on the validation
        - A _best_result_test file which contains a dictionary with the results, on the test set, of the best solution chosen using the validation set
    """

    parser = DataParser()
    seed = 1666
    URM_all = parser.get_URM_all()
    ICM_obj = parser.get_ICM_all()

    #URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.85, seed=seed)
    #URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.85, seed=seed)
    k = 5

    output_folder_path = "result_experiments_CV/"

    collaborative_algorithm_list = [
        HybridCombinationSearchCV
    ]

    from Base.Evaluation.Evaluator import EvaluatorHoldout

    combo_algorithm_list = [
        (ItemKNNCBFRecommender), {'topK': 164, 'shrink': 8, 'similarity': 'jaccard', 'normalize': True}
        (ItemKNNCBF_Special), {'topK': 1000, 'shrink': 1000, 'similarity': 'cosine', 'normalize': True, 'feature_weighting': 'BM25'},
        (ItemKNN_CBF_CF), {'topK': 1000, 'shrink': 1000, 'similarity': 'asymmetric', 'normalize': True, 'asymmetric_alpha': 0.241892724784089, 'feature_weighting': 'TF-IDF', 'icm_weight': 1.0},
        (ItemKNNCFRecommender), {'topK': 1000, 'shrink': 1000, 'similarity': 'cosine', 'normalize': True, 'feature_weighting': 'TF-IDF'},
        (UserKNNCFRecommender), {'topK': 163, 'shrink': 846, 'similarity': 'cosine', 'normalize': True, 'feature_weighting': 'TF-IDF'},
        (RP3betaRecommender), {'topK': 926, 'alpha': 0.4300109351916609, 'beta': 0.01807360750913967, 'normalize_similarity': False},
        (P3alphaRecommender), {'topK': 575, 'alpha': 0.48009885897470206, 'normalize_similarity': False},
        #(SLIM_BPR_Cython, {'topK': 989, 'epochs': 90, 'symmetric': False, 'sgd_mode': 'adam', 'lambda_i': 1.7432198130463203e-05, 'lambda_j': 0.0016819750046109673, 'learning_rate': 0.00031293205801039345})
    ]
    list_already_seen = []
    combinations_already_seen = combinations(list_already_seen, 3)
    """
    (icb, icf, p3a), (icb, icf, rp3b), (icb, icf, sen), (icb, p3a, rp3b), (icb, p3a, sen),
                                (icb, rp3b, sen), (icf, p3a, rp3b), (icf, p3a, sen)
    """
    combination_to_be_done = list(combinations(combo_algorithm_list, 3))

    for rec_perm in combination_to_be_done[:15]:

        if rec_perm not in combinations_already_seen:
            recommender_names = '_'.join([r[0].RECOMMENDER_NAME for r in rec_perm])
            output_folder_path = "result_experiments_CV2/seed_" + str(seed) + '/linear/' + recommender_names + '/'
            print(F"\nTESTING THE COMBO {recommender_names}")

            if (ItemKNNCBFRecommender not in rec_perm) and (ItemKNNCBF_Special not in rec_perm) and (ItemKNN_CBF_CF not in rec_perm) or (ItemKNNCBFRecommender in rec_perm and ItemKNNCBF_Special in rec_perm):
                # If directory does not exist, create
                if not os.path.exists(output_folder_path):
                    os.makedirs(output_folder_path)

                    runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                                       URM_train = URM_all,
                                                                       ICM_train = ICM_obj,
                                                                       metric_to_optimize = "MAP",
                                                                       n_cases = 50,
                                                                       n_random_starts = 20,
                                                                       output_folder_path = output_folder_path,
                                                                       parallelizeKNN = False,
                                                                       allow_weighting = False,
                                                                       k = k,
                                                                       seed = seed,
                                                                       list_rec = rec_perm,
                                                                       level = 'hybrid_search')
                    pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
                    pool.map(runParameterSearch_Collaborative_partial, collaborative_algorithm_list)


if __name__ == '__main__':
    read_data_split_and_search()
