from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Base.DataIO import DataIO
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython

from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from SLIM_ElasticNet.SSLIM_ElasticNet import SSLIMElasticNet


class LinearHybrid008(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "LinearHybrid008"

    # set the seed equal to the one of the parameter search!!!!
    def __init__(self, URM_train, ICM_train, submission=False, verbose=True, seed=1205):
        super(LinearHybrid008, self).__init__(URM_train, verbose=verbose)
        self.URM_train = URM_train
        self.ICM_train = ICM_train

        self.__rec1 = SSLIMElasticNet(URM_train, ICM_train, verbose=False)
        self.__rec1_params = {'beta': 0.4849594591575789, 'topK': 1000, 'l1_ratio': 1e-05, 'alpha': 0.001}

        self.__rec2 = ItemKNNCBFRecommender(URM_train, ICM_train, verbose=False)
        self.__rec2_params = {'topK': 65, 'shrink': 0, 'similarity': 'dice', 'normalize': True}

        self.__rec3 = UserKNNCFRecommender(URM_train, verbose=False)
        self.__rec3_params = {'topK': 190, 'shrink': 0, 'similarity': 'cosine', 'normalize': True}

        self.__a = self.__b = self.__c = None
        self.seed = seed
        self.__submission = submission

    def fit(self, alpha=0.5, l1_ratio=0.5):
        self.__a = alpha * l1_ratio
        self.__b = alpha - self.__a
        self.__c = 1 - self.__a - self.__b
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

            try:
                self.__rec2.load_model(f'stored_recommenders/seed_{str(self.seed)}_{self.__rec2.RECOMMENDER_NAME}/',
                                       f'best_for_{self.RECOMMENDER_NAME}')
                print(f"{self.__rec2.RECOMMENDER_NAME} loaded.")
            except:
                print(f"Fitting {self.__rec2.RECOMMENDER_NAME} ...")
                self.__rec2.fit(**self.__rec2_params)
                print(f"done.")
                self.__rec2.save_model(f'stored_recommenders/seed_{str(self.seed)}_{self.__rec2.RECOMMENDER_NAME}/',
                                       f'best_for_{self.RECOMMENDER_NAME}')

            try:
                self.__rec3.load_model(f'stored_recommenders/seed_{str(self.seed)}_{self.__rec3.RECOMMENDER_NAME}/',
                                       f'best_for_{self.RECOMMENDER_NAME}')
                print(f"{self.__rec3.RECOMMENDER_NAME} loaded.")
            except:
                print(f"Fitting {self.__rec3.RECOMMENDER_NAME} ...")
                self.__rec3.fit(**self.__rec3_params)
                print(f"done.")
                self.__rec3.save_model(f'stored_recommenders/seed_{str(self.seed)}_{self.__rec3.RECOMMENDER_NAME}/',
                                       f'best_for_{self.RECOMMENDER_NAME}')
        else:
            self.__rec1.fit(**self.__rec1_params)
            self.__rec2.fit(**self.__rec2_params)
            self.__rec3.fit(**self.__rec3_params)

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