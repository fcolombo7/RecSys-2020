#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""

from Base.NonPersonalizedRecommender import TopPop, Random
from EASE_R.EASE_R_Recommender import EASE_R_Recommender
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
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython
from MatrixFactorization.PureSVDRecommender import PureSVDRecommender, PureSVDItemRecommender
from MatrixFactorization.NMFRecommender import NMFRecommender

import traceback

import os, multiprocessing
from functools import partial


from DataParser import DataParser
from Data_manager.Movielens.Movielens10MReader import Movielens10MReader
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative, runParameterSearch_Content


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

    seed=1205
    parser = DataParser()

    URM_all = parser.get_URM_all()
    ICM_obj = parser.get_ICM_all()

    # SPLIT TO GET TEST PARTITION
    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.90, seed=seed)

    # SPLIT TO GET THE HYBRID VALID PARTITION
    URM_train, URM_valid_hybrid = split_train_in_two_percentage_global_sample(URM_train, train_percentage = 0.85, seed=seed)

    # SPLIT TO GET THE sub_rec VALID PARTITION
    URM_train, URM_valid_sub = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.85, seed=seed)

    output_folder_path = "result_experiments_v3/seed_"+str(seed)+'/'

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    collaborative_algorithm_list = [
        #EASE_R_Recommender
        #PipeHybrid001,
        #Random,
        #TopPop,
        #P3alphaRecommender,
        #RP3betaRecommender,
        #ItemKNNCFRecommender,
        #UserKNNCFRecommender,
        #MatrixFactorization_BPR_Cython,
        #MatrixFactorization_FunkSVD_Cython,
        #PureSVDRecommender,
        #NMFRecommender,
        #PureSVDItemRecommender
        #SLIM_BPR_Cython,
        SLIMElasticNetRecommender
        #IALSRecommender
        #MF_MSE_PyTorch
        #MergedHybrid000
        #LinearHybrid002
    ]

    content_algorithm_list= [
        #ItemKNNCBFRecommender
    ]


    from Base.Evaluation.Evaluator import EvaluatorHoldout

    evaluator_valid_sub = EvaluatorHoldout(URM_valid_sub, cutoff_list=[10])
    evaluator_valid_hybrid = EvaluatorHoldout(URM_valid_hybrid, cutoff_list=[10])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    """
        # TODO: setta I GIUSTI EVALUATOR QUI!!!!
    runParameterSearch_Content_partial = partial(runParameterSearch_Content,
                                                 URM_train=URM_train,
                                                 ICM_object=ICM_obj,
                                                 ICM_name='1BookFeatures',
                                                 n_cases = 50,
                                                 n_random_starts = 20,
                                                 evaluator_validation= evaluator_valid_sub,
                                                 evaluator_test = evaluator_valid_hybrid,
                                                 metric_to_optimize = "MAP",
                                                 output_folder_path=output_folder_path,
                                                 parallelizeKNN = False,
                                                 allow_weighting = True,
                                                 #similarity_type_list = ['cosine']
                                                 )
    pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
    pool.map(runParameterSearch_Content_partial, content_algorithm_list)
    """

    # TODO: setta I GIUSTI EVALUATOR QUI!!!!
    runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                       URM_train = URM_train,
                                                       ICM_train = ICM_obj,
                                                       metric_to_optimize = "MAP",
                                                       n_cases = 50,
                                                       n_random_starts = 20,
                                                       evaluator_validation_earlystopping = evaluator_valid_sub,
                                                       evaluator_validation = evaluator_valid_sub,
                                                       evaluator_test = evaluator_valid_hybrid,
                                                       output_folder_path = output_folder_path,
                                                       allow_weighting = False,
                                                       #similarity_type_list = ["cosine", 'jaccard'],
                                                       parallelizeKNN = False)
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
