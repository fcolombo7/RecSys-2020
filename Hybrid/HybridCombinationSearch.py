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

class HybridCombinationSearch(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "HybridCombinationSearch"

    # set the seed equal to the one of the parameter search!!!!
    def __init__(self, URM_train, ICM_train, list_rec = None, verbose=True, seed=1205):
        super(HybridCombinationSearch, self).__init__(URM_train, verbose = verbose)
        self.URM_train = URM_train
        self.ICM_train = ICM_train

        self.__rec1 = list_rec[0]
        self.__rec2 = list_rec[1]
        self.__rec3 = list_rec[2]

        self.__a = self.__b = self.__c = None
        self.seed=seed

    def fit(self, alpha=0.5, l1_ratio=0.5):
        self.__a = alpha * l1_ratio
        self.__b = alpha - self.__a
        self.__c = 1 - self.__a - self.__b
        
    def _compute_item_score(self, user_id_array, items_to_compute=None):

        item_weights_1 = self.__rec1._compute_item_score(user_id_array)
        item_weights_2 = self.__rec2._compute_item_score(user_id_array)
        item_weights_3 = self.__rec3._compute_item_score(user_id_array)

        #normalization
        item_weights_1_max = item_weights_1.max()
        item_weights_2_max = item_weights_2.max()
        item_weights_3_max = item_weights_3.max()

        if not item_weights_1_max == 0:
            item_weights_1 /= item_weights_1_max
        if not item_weights_2_max == 0:
            item_weights_2 /= item_weights_2_max
        if not item_weights_3_max == 0:
                item_weights_3 /= item_weights_3_max

        item_weights = item_weights_1 * self.__a + item_weights_2 * self.__b + item_weights_3 * self.__c

        return item_weights

    def save_model(self, folder_path, file_name = None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME
        self._print("Saving model in file '{}'".format(folder_path + file_name))
        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = {})
        self._print("Saving complete")


class HybridCombinationMergedSearch(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "HybridCombinationMergedSearch"

    # set the seed equal to the one of the parameter search!!!!
    def __init__(self, URM_train, ICM_train, list_rec=None, verbose=True, seed=1205):
        super(HybridCombinationMergedSearch, self).__init__(URM_train, verbose=verbose)
        self.URM_train = URM_train
        self.ICM_train = ICM_train

        self.__rec1 = list_rec[0]
        self.__rec2 = list_rec[1]
        self.__rec3 = list_rec[2]

        if (self.__rec1.W_sparse.shape != self.__rec2.W_sparse.shape) or \
                (self.__rec1.W_sparse.shape != self.__rec3.W_sparse.shape):
            raise ValueError(
                f"HybridCombinationMergedSearch({self.__rec1.RECOMMENDER_NAME} - {self.__rec2.RECOMMENDER_NAME} - "
                f"{self.__rec3.RECOMMENDER_NAME}): similarities have different size, S1 is "
                f"{self.__rec1.W_sparse.shape}, S2 is {self.__rec2.W_sparse.shape}, S3 is {self.__rec3.W_sparse.shape}")

        #self.__W1 = check_matrix(self.__rec1.W_sparse.copy(), 'csr')
        #self.__W2 = check_matrix(self.__rec2.W_sparse.copy(), 'csr')
        #self.__W3 = check_matrix(self.__rec3.W_sparse.copy(), 'csr')

        self.__a = self.__b = self.__c = None
        self.topK=None
        self.W_sparse=None
        self.seed = seed

    def fit(self, alpha=0.5, l1_ratio=0.5, topK=100):
        self.__a = alpha * l1_ratio
        self.__b = alpha - self.__a
        self.__c = 1 - self.__a - self.__b
        self.topK = topK

        W1_max = self.__rec1.W_sparse.max()
        W2_max = self.__rec2.W_sparse.max()
        W3_max = self.__rec3.W_sparse.max()

        W1 = self.__rec1.W_sparse
        W2 = self.__rec2.W_sparse
        W3 = self.__rec3.W_sparse
        if W1_max != 0:
            W1 = W1 / W1_max
        if W2_max != 0:
            W2 = W2 / W2_max
        if W3_max != 0:
            W3 = W3 / W3_max

        W = W1 * self.__a + W2 * self.__b + W3 * self.__c

        self.W_sparse = similarityMatrixTopK(W, k=self.topK).tocsr()

    def save_model(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME
        self._print("Saving model in file '{}'".format(folder_path + file_name))
        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save={})
        self._print("Saving complete")



