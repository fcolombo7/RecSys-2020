#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""

from Base.NonPersonalizedRecommender import TopPop, Random
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython
from MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from Hybrid.HybridLinear20Recommneder import HybridLinear20Recommneder
from EASE_R.EASE_R_Recommender import EASE_R_Recommender
from MatrixFactorization.IALSRecommender import IALSRecommender

import traceback

import os, multiprocessing
from functools import partial


from DataParser import DataParser
from Data_manager.Movielens.Movielens10MReader import Movielens10MReader
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative


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
    seed = 1024
    
    parser = DataParser()
    #dataReader = Movielens10MReader()
    #dataset = dataReader.load_data()

    URM_all = parser.get_URM_all()
    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.80, seed = seed)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage = 0.85, seed = seed)

    ranges = [(50, 100), (25, 50), (0, 25)]
    
    for ran in ranges:
        f_range = ran
        URM_validation = parser.filter_URM_test_by_range(URM_train, URM_validation, f_range)
        URM_test = parser.filter_URM_test_by_range(URM_train, URM_test, f_range)
        output_folder_path = "result_experiments/"+"range_"+str(f_range[0])+"-"+str(f_range[1])+"/"


        # If directory does not exist, create
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        collaborative_algorithm_list = [
            #Random,
            #TopPop,
            #P3alphaRecommender,
            #RP3betaRecommender,
            #ItemKNNCFRecommender,
            #UserKNNCFRecommender,
            #MatrixFactorization_BPR_Cython,
            #MatrixFactorization_FunkSVD_Cython,
            #PureSVDRecommender,
            SLIM_BPR_Cython,
            #SLIMElasticNetRecommender,
            #IALSRecommender
            #HybridLinear20Recommneder
            #EASE_R_Recommender
        ]

        if f_range in [(50, 100), (25, 50), (0, 25)]:
            collaborative_algorithm_list.append(MatrixFactorization_BPR_Cython)


        from Base.Evaluation.Evaluator import EvaluatorHoldout

        evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
        evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])


        runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                           URM_train = URM_train,
                                                           metric_to_optimize = "MAP",
                                                           n_cases = 60,
                                                           n_random_starts = 60*0.3,
                                                           evaluator_validation_earlystopping = evaluator_validation,
                                                           evaluator_validation = evaluator_validation,
                                                           evaluator_test = evaluator_test,
                                                           output_folder_path = output_folder_path,
                                                           #similarity_type_list = ["cosine"],
                                                           allow_weighting = True, #just added
                                                           parallelizeKNN = False)





        pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
        pool.map(runParameterSearch_Collaborative_partial, collaborative_algorithm_list)

        print("\n\n\n\n\n\n")
        print("#################################################################################")
        print(ran)
        print("#################################################################################")
        print("\n\n\n\n\n\n")
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
