import numpy as np
import scipy.sparse as sps

from sklearn.preprocessing import normalize
from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from GraphBased.RP3betaRecommender import RP3betaRecommender

from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
import time, sys

class RP3betaICM(RP3betaRecommender):

    RECOMMENDER_NAME = "RP3betaICM"

    def __init__(self, URM_train, ICM_train, verbose = True):
        super(RP3betaICM, self).__init__(URM_train, verbose = verbose)
        self.ICM_train = ICM_train
        
    def fit(self, icm_weight=1.0, **fit_args):

        ICM_train = self.ICM_train * icm_weight
        URM_train = sps.vstack([self.URM_train, self.ICM_train.T])
        self.URM_train = URM_train.tocsr()
        super(RP3betaICM, self).fit(**fit_args)