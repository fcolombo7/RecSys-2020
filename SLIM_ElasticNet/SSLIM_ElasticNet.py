import scipy.sparse as sps

from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender


class SSLIMElasticNet(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "S-SLIMElasticNet"

    def __init__(self, URM_train, ICM_train, verbose=True):
        super(SSLIMElasticNet, self).__init__(URM_train, verbose=verbose)
        self.URM_train = URM_train
        self.ICM_train = ICM_train
        self.__slim_rec = None
        self.beta = None
        self.W_sparse = None

    def fit(self, beta=0.5, l1_ratio=0.1, alpha=1.0, positive_only=True, topK=100):
        """
        beta is the parameter used to build the virtual URM that is given as input of the SLIM rec
        """
        self.beta = beta
        urm = self.URM_train * self.beta
        icm = self.ICM_train * (1 - self.beta)
        virtual_URM = sps.vstack([urm, icm.T])
        self.virtual_URM = virtual_URM.tocsr()
        self.__slim_rec = SLIMElasticNetRecommender(self.virtual_URM)
        self.__slim_rec.fit(l1_ratio=l1_ratio, alpha=alpha, positive_only=positive_only, topK=topK)
        self.W_sparse = self.__slim_rec.W_sparse