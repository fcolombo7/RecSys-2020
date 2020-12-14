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
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, \
    MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython


class HybridCombinationSearchCV(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "HybridCombinationSearchCV"

    # set the seed equal to the one of the parameter search!!!!
    def __init__(self, URM_train, ICM_train, list_rec, seed=None, fold=None, submission = False, verbose=True):
        """
        :params list_rec è una lista composta da (RecommenderClass, **fit_keywargs)
        """
        assert (seed is not None and fold is not None) or submission is True

        super(HybridCombinationSearchCV, self).__init__(URM_train, verbose=verbose)
        self.URM_train = URM_train
        self.ICM_train = ICM_train

        self.submission = submission
        self.seed = seed
        self.fold = fold
        self.list_rec = list_rec

        self.__rec1_class = list_rec[0][0]
        self.__rec1_keywargs = list_rec[0][1]

        self.__rec2_class = list_rec[1][0]
        self.__rec2_keywargs = list_rec[1][1]

        self.__rec3_class = list_rec[2][0]
        self.__rec3_keywargs = list_rec[2][1]

        ### CONSTRUCT the 3 recs
        try:
            self.__rec1 = self.__rec1_class(URM_train, ICM_train, verbose = False)
        except:
            self.__rec1 = self.__rec1_class(URM_train, verbose=False)
        try:
            self.__rec2 = self.__rec2_class(URM_train, ICM_train, verbose = False)
        except:
            self.__rec2 = self.__rec2_class(URM_train, verbose=False)
        try:
            self.__rec3 = self.__rec3_class(URM_train, ICM_train, verbose = False)
        except:
            self.__rec3 = self.__rec3_class(URM_train, verbose=False)

        self.__a = self.__b = self.__c = None
        self.seed = seed

    def fit(self, alpha=0.5, l1_ratio=0.5):
        self.__a = alpha * l1_ratio
        self.__b = alpha - self.__a
        self.__c = 1 - self.__a - self.__b

        folder = 'for_sub' if self.submission else 'hybrid_search'
        filename = 'fors_sub' if self.submission else f'{str(self.seed)}_fold-{str(self.fold)}'
        #load the models if already trained for that particular seed and fold
        try:
            self.__rec1.load_model(f'stored_recommenders/seed_{str(self.seed)}_{folder}/{self.__rec1.RECOMMENDER_NAME}/', filename)
            print(f"{self.__rec1.RECOMMENDER_NAME} loaded. [seed={self.seed}, fold={self.fold}]")
        except:
            print(f"Fitting {self.__rec1.RECOMMENDER_NAME} ... [seed={self.seed}, fold={self.fold}]")
            self.__rec1.fit(**self.__rec1_keywargs)
            print(f"done.")
            self.__rec1.save_model(f'stored_recommenders/seed_{str(self.seed)}_{folder}/{self.__rec1.RECOMMENDER_NAME}/', filename)

        try:
            self.__rec2.load_model(f'stored_recommenders/seed_{str(self.seed)}_{folder}/{self.__rec2.RECOMMENDER_NAME}/', filename)
            print(f"{self.__rec2.RECOMMENDER_NAME} loaded. [seed={self.seed}, fold={self.fold}]")
        except:
            print(f"Fitting {self.__rec2.RECOMMENDER_NAME} ... [seed={self.seed}, fold={self.fold}]")
            self.__rec2.fit(**self.__rec2_keywargs)
            print(f"done.")
            self.__rec2.save_model(f'stored_recommenders/seed_{str(self.seed)}_{folder}/{self.__rec2.RECOMMENDER_NAME}/', filename)

        try:
            self.__rec3.load_model(f'stored_recommenders/seed_{str(self.seed)}_{folder}/{self.__rec3.RECOMMENDER_NAME}/', filename)
            print(f"{self.__rec3.RECOMMENDER_NAME} loaded. [seed={self.seed}, fold={self.fold}]")
        except:
            print(f"Fitting {self.__rec3.RECOMMENDER_NAME} ... [seed={self.seed}, fold={self.fold}]")
            self.__rec3.fit(**self.__rec3_keywargs)
            print(f"done.")
            self.__rec3.save_model(f'stored_recommenders/seed_{str(self.seed)}_{folder}/{self.__rec3.RECOMMENDER_NAME}/', filename)

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        item_weights_1 = self.__rec1._compute_item_score(user_id_array)
        item_weights_2 = self.__rec2._compute_item_score(user_id_array)
        item_weights_3 = self.__rec3._compute_item_score(user_id_array)

        # normalization
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

    def save_model(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME
        self._print("Saving model in file '{}'".format(folder_path + file_name))
        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save={})
        self._print("Saving complete")