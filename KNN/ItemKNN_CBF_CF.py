
from Base.Recommender_utils import check_matrix
from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Base.IR_feature_weighting import okapi_BM_25, TF_IDF
import numpy as np
import scipy.sparse as sps

from Base.Similarity.Compute_Similarity import Compute_Similarity
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender


class ItemKNN_CBF_CF(ItemKNNCBFRecommender):
    RECOMMENDER_NAME = "ItemKNN_CBF_CF_Recommender"

    def fit(self, icm_weight=1.0, **fit_args):
        self.ICM_train_single = self.ICM_train.copy()

        ICM_train = self.ICM_train * icm_weight
        ICM_train = sps.hstack([ICM_train, self.URM_train.T])
        self.ICM_train = ICM_train.tocsr()
        super(ItemKNN_CBF_CF, self).fit(**fit_args)