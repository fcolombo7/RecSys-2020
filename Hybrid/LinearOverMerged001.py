from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Base.DataIO import DataIO
from Base.Recommender_utils import similarityMatrixTopK, check_matrix
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython

from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender


class LinearOverMerged001(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "LinearOverMerged001"

    # set the seed equal to the one of the parameter search!!!!
    def __init__(self, URM_train, ICM_train, submission=False, verbose=True, seed=1205):
        super(LinearOverMerged001, self).__init__(URM_train, verbose=verbose)
        self.URM_train = URM_train
        self.ICM_train = ICM_train
        self.__submission=submission
        self.__rec1 = UserKNNCFRecommender(URM_train, verbose=False)
        self.__rec1_params = {'topK': 190, 'shrink': 0, 'similarity': 'cosine', 'normalize': True}
        self.seed=seed

        icb = ItemKNNCBFRecommender(URM_train, ICM_train, verbose=False)
        icb_params = {'topK': 65, 'shrink': 0, 'similarity': 'dice', 'normalize': True}
        rp3b = RP3betaRecommender(URM_train, verbose=False)
        rp3b_params = {'topK': 1000, 'alpha': 0.38192761611274967, 'beta': 0.0, 'normalize_similarity': False}
        sen = SLIMElasticNetRecommender(URM_train, verbose=False)
        sen_params = {'topK': 992, 'l1_ratio': 0.004065081925341167, 'alpha': 0.003725005053334143}

        if not self.__submission:
            try:
                icb.load_model(f'stored_recommenders/seed_{str(self.seed)}_{icb.RECOMMENDER_NAME}/',
                               f'best_for_{self.RECOMMENDER_NAME}')
                print(f"{icb.RECOMMENDER_NAME} loaded.")
            except:
                print(f"Fitting {icb.RECOMMENDER_NAME} ...")
                icb.fit(**icb_params)
                print(f"done.")
                icb.save_model(f'stored_recommenders/seed_{str(self.seed)}_{icb.RECOMMENDER_NAME}/',
                               f'best_for_{self.RECOMMENDER_NAME}')
            try:
                rp3b.load_model(f'stored_recommenders/seed_{str(self.seed)}_{rp3b.RECOMMENDER_NAME}/',
                                f'best_for_{self.RECOMMENDER_NAME}')
                print(f"{rp3b.RECOMMENDER_NAME} loaded.")
            except:
                print(f"Fitting {rp3b.RECOMMENDER_NAME} ...")
                rp3b.fit(**rp3b_params)
                print(f"done.")
                rp3b.save_model(f'stored_recommenders/seed_{str(self.seed)}_{rp3b.RECOMMENDER_NAME}/',
                                f'best_for_{self.RECOMMENDER_NAME}')
            try:
                sen.load_model(f'stored_recommenders/seed_{str(self.seed)}_{sen.RECOMMENDER_NAME}/',
                               f'best_for_{self.RECOMMENDER_NAME}')
                print(f"{sen.RECOMMENDER_NAME} loaded.")
            except:
                print(f"Fitting {sen.RECOMMENDER_NAME} ...")
                sen.fit(**sen_params)
                print(f"done.")
                sen.save_model(f'stored_recommenders/seed_{str(self.seed)}_{sen.RECOMMENDER_NAME}/',
                               f'best_for_{self.RECOMMENDER_NAME}')
        else:
            icb.fit(**icb_params)
            rp3b.fit(**rp3b_params)
            sen.fit(**sen_params)

        self.__rec2 = HiddenMergedRecommender(URM_train, ICM_train, [icb, rp3b, sen], verbose=False)
        self.__rec2_params = {'alpha': 0.6355738550417837, 'l1_ratio': 0.6617849709204384, 'topK': 538}

        self.__a = self.__b = None

    def fit(self, alpha=0.5):
        self.__a = alpha
        self.__b = 1 - alpha
        if not self.__submission:
            try:
                self.__rec1.load_model(f'stored_recommenders/seed_{str(self.seed)}_{self.__rec1.RECOMMENDER_NAME}/',
                                       f'best_for_{self.RECOMMENDER_NAME}')
                print(f"{self.__rec1.RECOMMENDER_NAME} loaded.")
            except:
                print(f"Fitting {self.__rec1.RECOMMENDER_NAME} ...")
                self.__rec1.fit(**self.__rec1_params)
                print(f"done.")
                self.__rec1.save_model(f'stored_recommenders/seed_{str(self.seed)}_{self.__rec1.RECOMMENDER_NAME}/',
                                       f'best_for_{self.RECOMMENDER_NAME}')

        else:
            self.__rec1.fit(**self.__rec1_params)

        self.__rec2.fit(**self.__rec2_params)

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        item_weights_1 = self.__rec1._compute_item_score(user_id_array)
        item_weights_2 = self.__rec2._compute_item_score(user_id_array)

        item_weights = item_weights_1 * self.__a + item_weights_2 * self.__b

        return item_weights

    def save_model(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME
        self._print("Saving model in file '{}'".format(folder_path + file_name))
        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save={})
        self._print("Saving complete")


class HiddenMergedRecommender(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "LinearOverMerged001"

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

        self.__W1 = check_matrix(self.__rec1.W_sparse.copy(), 'csr')
        self.__W2 = check_matrix(self.__rec2.W_sparse.copy(), 'csr')
        self.__W3 = check_matrix(self.__rec3.W_sparse.copy(), 'csr')

        self.__a = self.__b = self.__c = None
        self.topK = None
        self.W_sparse = None
        self.seed = seed

    def fit(self, alpha=0.5, l1_ratio=0.5, topK=100):
        self.__a = alpha * l1_ratio
        self.__b = alpha - self.__a
        self.__c = 1 - self.__a - self.__b
        self.topK = topK

        W = self.__W1 * self.__a \
            + self.__W2 * self.__b \
            + self.__W3 * self.__c

        self.W_sparse = similarityMatrixTopK(W, k=self.topK).tocsr()

    def save_model(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME
        self._print("Saving model in file '{}'".format(folder_path + file_name))
        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save={})
        self._print("Saving complete")