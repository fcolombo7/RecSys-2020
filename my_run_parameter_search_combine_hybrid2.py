#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""

from Base.NonPersonalizedRecommender import TopPop, Random
from EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Hybrid.HybridCombinationSearch import HybridCombinationSearch
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

from ParameterTuning.my_run_parameter_search import runParameterSearch_Collaborative, runParameterSearch_Content


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

    seed = 1205
    parser = DataParser()

    URM_all = parser.get_URM_all()
    ICM_obj = parser.get_ICM_all()

    # SPLIT TO GET TEST PARTITION
    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.90, seed=seed)

    # SPLIT TO GET THE HYBRID VALID PARTITION
    URM_train, URM_valid_hybrid = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.85,
                                                                              seed=seed)

    URM_valid_hybrid = parser.filter_URM_test_by_range(URM_train, URM_valid_hybrid, (3, -1))

    collaborative_algorithm_list = [
        # EASE_R_Recommender
        # PipeHybrid001,
        # Random,
        # TopPop,
        # P3alphaRecommender,
        # RP3betaRecommender,
        # ItemKNNCFRecommender,
        # UserKNNCFRecommender,
        # MatrixFactorization_BPR_Cython,
        # MatrixFactorization_FunkSVD_Cython,
        # PureSVDRecommender,
        # NMFRecommender,
        # PureSVDItemRecommender
        # SLIM_BPR_Cython,
        # SLIMElasticNetRecommender
        # IALSRecommender
        # MF_MSE_PyTorch
        # MergedHybrid000
        # LinearHybrid002ggg
        HybridCombinationSearch
    ]

    content_algorithm_list = [
        # ItemKNNCBFRecommender
    ]

    from Base.Evaluation.Evaluator import EvaluatorHoldout

    evaluator_valid_hybrid = EvaluatorHoldout(URM_valid_hybrid, cutoff_list=[10])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    """
    earlystopping_keywargs = {"validation_every_n": 5,
                              "stop_on_validation": True,
                              "evaluator_object": evaluator_valid_hybrid,
                              "lower_validations_allowed": 5,
                              "validation_metric": 'MAP',
                              }
    
    print('IALS training...')
    ials = IALSRecommender(URM_train, verbose=False)
    ials_params = {'num_factors': 83, 'confidence_scaling': 'linear', 'alpha': 28.4278070726612,
                   'epsilon': 1.0234211788885077, 'reg': 0.0027328110246575004, 'epochs': 20}
    ials.fit(**ials_params, **earlystopping_keywargs)
    print("Done")
    
    
    print("PureSVD training...")
    psvd = PureSVDRecommender(URM_train, verbose=False)
    psvd_params = {'num_factors': 711}
    psvd.fit(**psvd_params)
    print("Done")
    """
    print("Rp3beta training...")
    rp3b = RP3betaRecommender(URM_train, verbose=False)
    rp3b_params = {'topK': 753, 'alpha': 0.3873710051288722, 'beta': 0.0, 'normalize_similarity': False}
    rp3b.fit(**rp3b_params)
    print("Done")
    print("P3alpha training...")
    p3a = P3alphaRecommender(URM_train, verbose=False)
    p3a_params = {'topK': 438, 'alpha': 0.41923120471415165, 'normalize_similarity': False}
    p3a.fit(**p3a_params)
    print("Done")
    print("ItemKnnCF training...")
    icf = ItemKNNCFRecommender(URM_train, verbose=False)
    icf_params = {'topK': 565, 'shrink': 554, 'similarity': 'tversky', 'normalize': True, 'tversky_alpha': 1.9109121434662428, 'tversky_beta': 1.7823834698905734}
    icf.fit(**icf_params)
    print("Done")
    print("UserKnnCF training...")
    ucf = UserKNNCFRecommender(URM_train, verbose=False)
    ucf_params = {'topK': 190, 'shrink': 0, 'similarity': 'cosine', 'normalize': True}
    ucf.fit(**ucf_params)
    print("Done")
    print("ItemKnnCBF training...")
    icb = ItemKNNCBFRecommender(URM_train, ICM_obj, verbose=False)
    icb_params = {'topK': 205, 'shrink': 1000, 'similarity': 'cosine', 'normalize': True, 'feature_weighting': 'BM25'}
    icb.fit(**icb_params)
    print("Done")
    """
    print("SlimElasticNet training...")
    sen = SLIMElasticNetRecommender(URM_train, verbose=False)
    sen_params = {'topK': 954, 'l1_ratio': 3.87446082207643e-05, 'alpha': 0.07562657698792305}
    sen.fit(**sen_params)
    print("Done")
    """

    list_recommender = [icb, icf, ucf, p3a, rp3b]
    list_already_seen = []

    for rec_perm in combinations(list_recommender, 3):

        if rec_perm not in combinations(list_already_seen, 3):

            recommender_names = '_'.join([r.RECOMMENDER_NAME for r in rec_perm])
            output_folder_path = "result_experiments_v3/seed_" + str(seed)+'_3--1' + '/' + recommender_names + '/'

            # If directory does not exist, create
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)

            # TODO: setta I GIUSTI EVALUATOR QUI!!!!
            runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                               URM_train=URM_train,
                                                               ICM_train=ICM_obj,
                                                               metric_to_optimize="MAP",
                                                               n_cases=50,
                                                               n_random_starts=20,
                                                               evaluator_validation_earlystopping=evaluator_valid_hybrid,
                                                               evaluator_validation=evaluator_valid_hybrid,
                                                               evaluator_test=evaluator_test,
                                                               output_folder_path=output_folder_path,
                                                               allow_weighting=False,
                                                               # similarity_type_list = ["cosine", 'jaccard'],
                                                               parallelizeKNN=False,
                                                               list_rec=rec_perm)
            pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
            pool.map(runParameterSearch_Collaborative_partial, collaborative_algorithm_list)

    #
    #
    # for recommender_class in collaborative_algorithm_list:
    #
    #     try:
    #
    #         runParameterSearch_Collaborative_partial(recommender_class)
    #
    #     except Exception as e:
    #
    #         print("On recommender {} Exception {}".format(recommender_class, str(e)))
    #         traceback.print_exc()
    #


if __name__ == '__main__':
    read_data_split_and_search()