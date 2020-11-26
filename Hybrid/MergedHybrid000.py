from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender

from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Base.Recommender_utils import check_matrix, similarityMatrixTopK

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample


class MergedHybrid000(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "MergedHybrid000"

    def __init__(self, URM_train, verbose=True):
        super(MergedHybrid000, self).__init__(URM_train, verbose = verbose)

        rec1 = RP3betaRecommender(URM_train, verbose=False)
        try:
            rec1.load_model('stored_recommenders/RP3betaRecommender/best_at_26_10_20')
        except:
            rec1.fit(alpha=0.4530815441932864,  beta=0.008742088319964482, topK=104, normalize_similarity=False)
            rec1.save_model('stored_recommenders/RP3betaRecommender/best_at_26_10_20')

        rec2 = ItemKNNCFRecommender(URM_train, verbose=False)
        try:
            rec2.load_model('stored_recommenders/ItemKNNCFRecommender/best_at_26_10_20')
        except:
            rec2.fit(topK=967, shrink=356, similarity='cosine', normalize=True)
            rec2.save_model('stored_recommenders/ItemKNNCFRecommender/best_at_26_10_20')

        self.rec1 = rec1
        self.rec2 = rec2
        self.rec1_W_sparse = rec1.W_sparse.copy()
        self.rec2_W_sparse = rec2.W_sparse.copy()
        self.URM_train = URM_train
        #self._URM_train_format_checked = False
        #self._W_sparse_format_checked = False

    def fit(self, alpha=0.5, topK=100):
        self.alpha = alpha
        self.topK = topK
        W = self.rec1_W_sparse*self.alpha + self.rec2_W_sparse*(1-self.alpha)
        self.W_sparse = similarityMatrixTopK(W, k=self.topK).tocsr()
