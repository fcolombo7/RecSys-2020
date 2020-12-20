import numpy as np
import scipy.sparse as sps

from sklearn.preprocessing import normalize
from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from GraphBased.P3alphaRecommender import P3alphaRecommender

from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
import time, sys

class P3alphaICM(P3alphaRecommender):

    RECOMMENDER_NAME = "P3alphaICM"

    def __init__(self, URM_train, ICM_train, verbose = True):
        super(P3alphaICM, self).__init__(URM_train, verbose = verbose)
        self.ICM_train = ICM_train
        
    def fit(self, icm_weight=1.0, **fit_args):

        ICM_train = self.ICM_train * icm_weight
        URM_train = sps.vstack([self.URM_train, self.ICM_train.T])
        self.URM_train = URM_train.tocsr()
        super(P3alphaICM, self).fit(**fit_args)