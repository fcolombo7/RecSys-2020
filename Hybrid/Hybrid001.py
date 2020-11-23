from Base.BaseRecommender import BaseRecommender
from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Base.Recommender_utils import check_matrix

class Hybrid001(BaseRecommender):

    def __init__(self, URM_train, content_recommender: ItemKNNCBFRecommender, collaborative_recommender: SLIMElasticNetRecommender):
        self.content_W_sparse = content_recommender.W_sparse
        self.collaborative_W_sparse = collaborative_recommender.W_sparse
        self.URM_train = URM_train
        self._URM_train_format_checked = False
        self._W_sparse_format_checked = False

    def fit(self, alpha):
        self.alpha = alpha
        self.W_sparse = self.content_W_sparse.multiply(alpha) + self.collaborative_W_sparse.multiply((1-alpha))

    def recommend(self, user_id_array, cutoff = None, remove_seen_flag=True, items_to_compute = None,
                  remove_top_pop_flag = False, remove_custom_items_flag = False, return_scores = False):
        raise NotImplementedError