from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_AsySVD_Cython
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Base.DataIO import DataIO
import numpy as np
import operator
import os
import scipy.sparse as sps



class HybridLinear10Recommneder(BaseItemSimilarityMatrixRecommender):

    RECOMMENDER_NAME = "HybridLinear10Recommender"

    def __init__(self, URM_train, seed: int):
        super(HybridLinear10Recommneder, self).__init__(URM_train)
        self.slimBPR = SLIM_BPR_Cython(URM_train)
        self.userKnnCF = UserKNNCFRecommender(URM_train)
        #self.itemcf = ItemKNNCFRecommender(urm)

    def fit(self, alpha=1):
        self.slimBPR.fit(epochs= 135, topK= 933, symmetric= False, sgd_mode= 'adagrad', lambda_i= 1.054e-05, lambda_j= 1.044e-05, learning_rate= 0.00029)
        self.userKnnCF.fit(topK= 201, shrink= 998, similarity= 'cosine', normalize= True, feature_weighting= 'TF-IDF')
        self.alpha = alpha
        self.beta = 1 - alpha
        #self.gamma = alpha_gamma_ratio

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        # ATTENTION!
        # THIS METHOD WORKS ONLY IF user_id_array IS A SCALAR AND NOT AN ARRAY
        # TODO

        scores_slimBPR = self.slimBPR._compute_item_score(user_id_array=user_id_array)
        scores_userKnnCF = self.userKnnCF._compute_item_score(user_id_array=user_id_array)

        # normalization
        #slim_max = scores_slim.max()
        #rp3_max = scores_rp3.max()
        #itemcf_max = scores_itemcf.max()

        #if not slim_max == 0:
         #   scores_slim /= slim_max
        #if not rp3_max == 0:
         #   scores_rp3 /= rp3_max
        #if not itemcf_max == 0:
         #   scores_itemcf /= itemcf_max

        scores_total = self.alpha * scores_slimBPR + self.beta * scores_userKnnCF #+ self.gamma * scores_itemcf

        return scores_total
    
    def save_model(self, folder_path, file_name = None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))


        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = {})

        self._print("Saving complete")