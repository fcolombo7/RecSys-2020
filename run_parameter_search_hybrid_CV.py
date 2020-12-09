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

from ParameterTuningCV.my_run_parameter_search import runParameterSearch_Collaborative


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

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.85, seed=seed)
    #URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.85, seed=seed)
    k = 5

    output_folder_path = "result_experiments_CV/"

    collaborative_algorithm_list = [
        HybridCombinationSearchCV
    ]

    from Base.Evaluation.Evaluator import EvaluatorHoldout

    combo_algorithm_list = [
        (ItemKNNCBFRecommender, {'topK': 22, 'shrink': 59, 'similarity': 'dice', 'normalize': False}),
        (ItemKNNCFRecommender, {'topK': 994, 'shrink': 981, 'similarity': 'asymmetric', 'normalize': True, 'asymmetric_alpha': 0.05110465631417439, 'feature_weighting': 'TF-IDF'})
    ]
    list_already_seen = []
    combinations_already_seen = combinations(list_already_seen, 3)
    """
    (icb, icf, p3a), (icb, icf, rp3b), (icb, icf, sen), (icb, p3a, rp3b), (icb, p3a, sen),
                                (icb, rp3b, sen), (icf, p3a, rp3b), (icf, p3a, sen)
    """

    for rec_perm in combinations(combo_algorithm_list, 3):

        if rec_perm not in combinations_already_seen:
            recommender_names = '_'.join([r[0].RECOMMENDER_NAME for r in rec_perm])
            output_folder_path = "result_experiments_CV/seed_" + str(seed) + '/linear_combination/' + recommender_names + '/'
            print(F"\nTESTING THE COMBO {recommender_names}")

            # If directory does not exist, create
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)

            runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                               URM_train = URM_train,
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
