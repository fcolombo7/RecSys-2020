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



class Hybrid110Recommender(BaseItemSimilarityMatrixRecommender):

    RECOMMENDER_NAME = "Hybrid110Recommender"

    def __init__(self, URM_train, seed: int):
        super(Hybrid110Recommender, self).__init__(URM_train)
        self.number_of_interactions_per_user = (self.URM_train > 0).sum(axis=1)
        self.highRange = SLIM_BPR_Cython(URM_train)
        self.lowRange = P3alphaRecommender(URM_train)
        self.midRange = RP3betaRecommender(URM_train)
        #self.itemcf = ItemKNNCFRecommender(urm)

    def fit(self):
        try:
            self.highRange.load_model("result_experiments/range_200--1/","SLIM_BPR_Recommender_best_model")
        except:
            self.highRange.fit(topK= 100, epochs= 70, symmetric= False, sgd_mode= 'adam', lambda_i= 0.01, lambda_j= 1e-05, learning_rate= 0.0001)
        self.lowRange.fit(topK= 685, alpha= 0.41303525095465676, normalize_similarity= False)
        self.midRange.fit(topK= 979, alpha= 0.42056182126095865, beta= 0.03446674275249296, normalize_similarity= False)
        
        #self.gamma = alpha_gamma_ratio
        
    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True,  items_to_compute=None,
                  remove_top_pop_flag=False, remove_custom_items_flag = False, return_scores=False):
        
        if user_id_array in self._user_with_interactions_within(0, 50):
                return self.lowRange.recommend(user_id_array, cutoff=cutoff)
        elif user_id_array in self._user_with_interactions_within(51, 200):
                return self.midRange.recommend(user_id_array, cutoff=cutoff)
        elif user_id_array in self._user_with_interactions_over(200):
                return self.highRange.recommend(user_id_array, cutoff=cutoff)
        """recs = []
        for user_id in user_id_array:
            if user_id in self._user_with_interactions_within(0, 200):
                rec = self.lowRange.recommend(user_id_array, cutoff=cutoff)
            elif user_id in self._user_with_interactions_over(200):
                rec = self.highRange.recommend(user_id_array, cutoff=cutoff)
            recs.append(rec)
        scores_batch = self._compute_item_score(user_id_array, items_to_compute=items_to_compute)
        return np.array(recs), scores_batch"""
    
    def _user_with_interactions_within(self, x=0, y=200):
        a = self.number_of_interactions_per_user
        x1 = np.where(a >= x)
        x2 = np.where(a <= y)
        return np.array([n for n in x1[0] if n in x2[0]])
    
    def _user_with_interactions_over(self, x=0):
        a = self.number_of_interactions_per_user
        x1 = np.where(a > x)
        return np.array([n for n in x1[0]])
    
    def save_model(self, folder_path, file_name = None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))


        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = {})

        self._print("Saving complete")