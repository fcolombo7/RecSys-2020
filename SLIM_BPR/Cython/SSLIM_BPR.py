from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
import scipy.sparse as sps


class SSLIM_BPR(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "S-SLIM_BPR"

    def __init__(self, URM_train, ICM_train, verbose = True, free_mem_threshold = 0.5, recompile_cython = False):
        super(SSLIM_BPR, self).__init__(URM_train, verbose = verbose)
        self.ICM_train = ICM_train
        self.free_mem_treshold = free_mem_threshold
        self.recompile_cython = recompile_cython
        self.verbose = verbose

    def fit(self,
            alpha = 0.5,
            epochs=300,
            positive_threshold_BPR = None,
            train_with_sparse_weights = None,
            symmetric = True,
            random_seed = None,
            batch_size = 1000, lambda_i = 0.0, lambda_j = 0.0, learning_rate = 1e-4, topK = 200,
            sgd_mode='adagrad', gamma=0.995, beta_1=0.9, beta_2=0.999,
            **earlystopping_kwargs):

        # build the virtual URM
        self.alpha = alpha
        urm = self.URM_train * self.alpha
        icm = self.ICM_train * (1 - self.alpha)
        virtual_URM = sps.vstack([urm, icm.T])
        self.virtual_URM = virtual_URM.tocsr()

        self.__slim_bpr = SLIM_BPR_Cython(self.virtual_URM, verbose=self.verbose, free_mem_threshold=self.free_mem_treshold, recompile_cython=self.recompile_cython)
        self.__slim_bpr.fit(epochs=epochs,
                            positive_threshold_BPR=positive_threshold_BPR,
                            train_with_sparse_weights=train_with_sparse_weights,
                            symmetric=symmetric,
                            random_seed=random_seed,
                            batch_size=batch_size, lambda_i=lambda_i, lambda_j=lambda_j, learning_rate=learning_rate, topK=topK,
                            sgd_mode=sgd_mode, gamma=gamma, beta_1=beta_1, beta_2=beta_2,
                            **earlystopping_kwargs
                            )
        self.W_sparse = self.__slim_bpr.W_sparse
