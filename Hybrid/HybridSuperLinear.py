from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Base.DataIO import DataIO
from Base.Recommender_utils import check_matrix, similarityMatrixTopK

from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from EASE_R.EASE_R_Recommender import EASE_R_Recommender

# KNN machine learning
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender

# Matrix Factorization
from MatrixFactorization.PureSVDRecommender import PureSVDRecommender, PureSVDItemRecommender
from MatrixFactorization.IALSRecommender import IALSRecommender
from MatrixFactorization.NMFRecommender import NMFRecommender
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython,\
    MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
import numpy as np

class HybridSuperLinear(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "HybridSuperLinear"

    # set the seed equal to the one of the parameter search!!!!
    def __init__(self, URM_train, ICM_train, list_rec = None, verbose=True, seed=1205):
        super(HybridSuperLinear, self).__init__(URM_train, verbose = verbose)
        self.URM_train = URM_train
        self.ICM_train = ICM_train

        self.__rec1 = list_rec[0]
        self.__rec2 = list_rec[1]
        self.__rec3 = list_rec[2]
        self.__rec4 = list_rec[3]
        self.__rec5 = list_rec[4]
        self.__rec6 = list_rec[5]
        self.__rec7 = list_rec[6]
        self.__rec8 = list_rec[7]

        self.__a = self.__b = self.__c = self.__d = self.__e = self.__f = self.__g = self.__h = None
        self.seed=seed

    def fit(self, a, b, c, d, e, f, g, h, norm = True):
        if norm:
            params = [a, b, c, d, e, f, g, h]
            sum_p = np.sum(params)
        else: sum_p = 1
        self.__a = a / sum_p
        self.__b = b / sum_p
        self.__c = c / sum_p
        self.__d = d / sum_p
        self.__e = e / sum_p
        self.__f = f / sum_p
        self.__g = g / sum_p
        self.__h = h / sum_p
        
    def _compute_item_score(self, user_id_array, items_to_compute=None):

        item_weights_1 = self.__rec1._compute_item_score(user_id_array)
        item_weights_2 = self.__rec2._compute_item_score(user_id_array)
        item_weights_3 = self.__rec3._compute_item_score(user_id_array)
        item_weights_4 = self.__rec4._compute_item_score(user_id_array)
        item_weights_5 = self.__rec5._compute_item_score(user_id_array)
        item_weights_6 = self.__rec6._compute_item_score(user_id_array)
        item_weights_7 = self.__rec7._compute_item_score(user_id_array)
        item_weights_8 = self.__rec8._compute_item_score(user_id_array)

        #normalization
        item_weights_1_max = item_weights_1.max()
        item_weights_2_max = item_weights_2.max()
        item_weights_3_max = item_weights_3.max()
        item_weights_4_max = item_weights_4.max()
        item_weights_5_max = item_weights_5.max()
        item_weights_6_max = item_weights_6.max()
        item_weights_7_max = item_weights_7.max()
        item_weights_8_max = item_weights_8.max()

        if not item_weights_1_max == 0:
            item_weights_1 /= item_weights_1_max
        if not item_weights_2_max == 0:
            item_weights_2 /= item_weights_2_max
        if not item_weights_3_max == 0:
            item_weights_3 /= item_weights_3_max
        if not item_weights_4_max == 0:
            item_weights_4 /= item_weights_4_max
        if not item_weights_5_max == 0:
            item_weights_5 /= item_weights_5_max
        if not item_weights_6_max == 0:
            item_weights_6 /= item_weights_6_max
        if not item_weights_7_max == 0:
            item_weights_7 /= item_weights_7_max
        if not item_weights_8_max == 0:
            item_weights_8 /= item_weights_8_max
        
        item_weights = item_weights_1 * self.__a + item_weights_2 * self.__b + item_weights_3 * self.__c + item_weights_4 * self.__d + item_weights_5 * self.__e + item_weights_6 * self.__f + item_weights_7 * self.__g + item_weights_8 * self.__h
        
        return item_weights

    def save_model(self, folder_path, file_name = None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME
        self._print("Saving model in file '{}'".format(folder_path + file_name))
        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = {})
        self._print("Saving complete")
