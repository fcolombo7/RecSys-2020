from Base.BaseRecommender import BaseRecommender
import numpy as np

from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Base.DataIO import DataIO
from Base.Recommender_utils import similarityMatrixTopK

from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from Hybrid.HybridCombinationSearchCV import HybridCombinationSearchCV
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNN_CBF_CF import ItemKNN_CBF_CF
from KNN.SpecialItemKNNCBFRecommender import SpecialItemKNNCBFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SSLIM_ElasticNet import SSLIMElasticNet


class UserWiseRecommenderCV(BaseRecommender):
    RECOMMENDER_NAME = "UserWiseRecommenderCV"

    def __init__(self, URM_train, ICM_train, rec_range_list, seed=None, fold=None, submission=False, verbose=True,
                 name=None):
        """
        :params rec_range_list: lista contenente tuple formate nel seguente modo: pos0 = (start_range, end_range),
        pos1 = ['sigla_rec1', 'sigla_rec2', 'sigla_rec3'] #lista di sigle, oppure singola sigla 'sigla_rec'
        pos2 = **params
        """
        assert (seed is not None and fold is not None) or submission is True
        super(UserWiseRecommenderCV, self).__init__(URM_train, verbose=verbose)

        self.seed = seed
        self.fold = fold
        self.submission = submission

        self.URM_train=URM_train
        self.ICM_train=ICM_train
        self.rec_range_list = rec_range_list
        if name is not None:
            self.RECOMMENDER_NAME = name

        self.__recommenders_dict = {
            'icb': (ItemKNNCBFRecommender, {'topK': 164, 'shrink': 8, 'similarity': 'jaccard', 'normalize': True}),
            'icbsup': (SpecialItemKNNCBFRecommender, {'topK': 1000, 'shrink': 1000, 'similarity': 'cosine',
                                                      'normalize': True, 'feature_weighting': 'BM25'}),
            'icfcb': (ItemKNN_CBF_CF, {'topK': 1000, 'shrink': 1000, 'similarity': 'asymmetric', 'normalize': True,
                                       'asymmetric_alpha': 0.241892724784089, 'feature_weighting': 'TF-IDF',
                                       'icm_weight': 1.0}),
            'icf': (ItemKNNCFRecommender, {'topK': 1000, 'shrink': 1000, 'similarity': 'cosine', 'normalize': True,
                                           'feature_weighting': 'TF-IDF'}),
            'ucf': (UserKNNCFRecommender, {'topK': 163, 'shrink': 846, 'similarity': 'cosine', 'normalize': True,
                                           'feature_weighting': 'TF-IDF'}),
            'p3a': (RP3betaRecommender, {'topK': 926, 'alpha': 0.4300109351916609, 'beta': 0.01807360750913967,
                                         'normalize_similarity': False}),
            'rp3b': (P3alphaRecommender, {'topK': 575, 'alpha': 0.48009885897470206, 'normalize_similarity': False}),
            'sbpr': (SLIM_BPR_Cython,
                     {'topK': 1000, 'epochs': 130, 'symmetric': False, 'sgd_mode': 'adam', 'lambda_i': 1e-05,
                      'lambda_j': 1e-05, 'learning_rate': 0.0001}),
            'sslim': (SSLIMElasticNet, {'beta': 0.567288665094892, 'topK': 1000, 'l1_ratio': 1e-05, 'alpha': 0.001}),
        }

        self.__recommender_segmentation = []

    def fit(self):
        for f_range, rec_reference, params in self.rec_range_list:
            if isinstance(rec_reference, list):
                print(f"> Range {str(f_range)} [Hybrid]")
                temp_list = []
                for rec in rec_reference:
                    temp_list.append(self.__recommenders_dict[rec])
                h_rec = HybridCombinationSearchCV(self.URM_train, self.ICM_train, temp_list, seed=self.seed, fold=self.fold, submission=self.submission, verbose=False)
                h_rec.fit(**params)
                self.__recommender_segmentation.append((f_range, h_rec))
            else:
                print(f"> Range {str(f_range)} [Single]")
                s_rec = None
                try:
                    s_rec1 = self.__recommenders_dict[rec_reference][0](self.URM_train, self.ICM_train, verbose=False)
                except:
                    s_rec1 = self.__recommenders_dict[rec_reference][0](self.URM_train, verbose=False)
                folder = 'for_sub' if self.submission else 'hybrid_search'
                filename = 'fors_sub' if self.submission else f'{str(self.seed)}_fold-{str(self.fold)}'
                try:
                    s_rec1.load_model(f'stored_recommenders/seed_{str(self.seed)}_{folder}/{s_rec1.RECOMMENDER_NAME}/',
                            filename)
                    print(f"{s_rec1.RECOMMENDER_NAME} loaded. [seed={self.seed}, fold={self.fold}]")
                except:
                    print(f"Fitting {s_rec1.RECOMMENDER_NAME} ... [seed={self.seed}, fold={self.fold}]")
                    s_rec1.fit(**self.__recommenders_dict[rec_reference][1])
                    print(f"done.")
                    s_rec1.save_model(
                            f'stored_recommenders/seed_{str(self.seed)}_{folder}/{s_rec1.RECOMMENDER_NAME}/',
                            filename)

                self.__recommender_segmentation.append((f_range, s_rec))

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_custom_items_flag=False, return_scores=False):
        # If is a scalar transform it in a 1-cell array
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False

        if cutoff is None:
            cutoff = self.URM_train.shape[1] - 1

        ranking_list = []
        for user_id in user_id_array:
            user_profile_length = len(self.URM_train[user_id].data)
            recommender = self.__get_recommender_by_profile_length(user_profile_length)
            ranking_list.append(recommender.recommend(user_id,
                                                      cutoff=cutoff,
                                                      remove_seen_flag=remove_seen_flag,
                                                      items_to_compute=items_to_compute,
                                                      remove_top_pop_flag=remove_top_pop_flag,
                                                      remove_custom_items_flag=remove_custom_items_flag,
                                                      return_scores=False)
                                )
        # Return single list for one user, instead of list of lists
        if single_user:
            ranking_list = ranking_list[0]
        return ranking_list

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        raise NotImplementedError(
            f"{self.RECOMMENDER_NAME}: compute_item_score not assigned for current recommender, unable to compute "
            f"prediction scores")

    def save_model(self, folder_path, file_name=None):
        raise NotImplementedError(
            f"{self.RECOMMENDER_NAME}: save_model not assigned for current recommender, unable to save the model")

    def __get_recommender_by_profile_length(self, user_profile_length):
        for f_range, recommender in self.__recommender_segmentation:
            if user_profile_length >= f_range[0] and (user_profile_length < f_range[1] or f_range[1] == -1):
                # print (f_range, recommender.RECOMMENDER_NAME)
                return recommender
        raise ValueError(
            f"{self.RECOMMENDER_NAME}: there is no recommender for users with profile length equal to {user_profile_length}.")

"""
class HiddenLinearRecommender(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "HiddenRecommender"

    # set the seed equal to the one of the parameter search!!!!
    def __init__(self, URM_train, ICM_train, rec_list, submission=False, verbose=True, seed=1205):
        super(HiddenLinearRecommender, self).__init__(URM_train, verbose=verbose)
        self.URM_train = URM_train
        self.ICM_train = ICM_train

        self.__rec1 = rec_list[0]
        self.__rec2 = rec_list[1]
        self.__rec3 = rec_list[2]

        self.__a = self.__b = self.__c = None
        self.seed = seed
        self.__submission = submission

    def fit(self, alpha=0.5, l1_ratio=0.5):
        self.__a = alpha * l1_ratio
        self.__b = alpha - self.__a
        self.__c = 1 - self.__a - self.__b

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
class HiddenMergedRecommender(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "HiddenMergedRecommender"

    # set the seed equal to the one of the parameter search!!!!
    def __init__(self, URM_train, ICM_train, rec_list, submission=False, verbose=True, seed=1205):
        super(HiddenMergedRecommender, self).__init__(URM_train, verbose=verbose)
        self.URM_train = URM_train
        self.ICM_train = ICM_train

        self.__rec1 = rec_list[0]
        self.__rec2 = rec_list[1]
        self.__rec3 = rec_list[2]

        if (self.__rec1.W_sparse.shape != self.__rec2.W_sparse.shape) or \
                (self.__rec1.W_sparse.shape != self.__rec3.W_sparse.shape):
            raise ValueError(
                f"HybridCombinationMergedSearch({self.__rec1.RECOMMENDER_NAME} - {self.__rec2.RECOMMENDER_NAME} - "
                f"{self.__rec3.RECOMMENDER_NAME}): similarities have different size, S1 is "
                f"{self.__rec1.W_sparse.shape}, S2 is {self.__rec2.W_sparse.shape}, S3 is {self.__rec3.W_sparse.shape}")

        # self.__W1 = check_matrix(self.__rec1.W_sparse.copy(), 'csr')
        # self.__W2 = check_matrix(self.__rec2.W_sparse.copy(), 'csr')
        # self.__W3 = check_matrix(self.__rec3.W_sparse.copy(), 'csr')

        self.__a = self.__b = self.__c = None
        self.topK = None
        self.W_sparse = None
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
"""