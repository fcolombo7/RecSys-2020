"""
@author: Massimo Quadrana
"""


import numpy as np
import scipy.sparse as sps

from Base.Recommender_utils import check_matrix
from sklearn.linear_model import ElasticNet
from sklearn.exceptions import ConvergenceWarning

from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit
import time, sys, warnings


class SSLIMElasticNet(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "S-SLIMElasticNet"

    def __init__(self, URM_train, ICM_train, verbose=True):
        super(SSLIMElasticNet, self).__init__(URM_train, verbose=verbose)
        self.URM_train = URM_train
        self.ICM_train = ICM_train
        self.__slim_rec = None
        self.beta = None
        self.W_sparse = None

    """
    def fit(self, beta=0.5, l1_ratio=0.1, alpha=1.0, positive_only=True, topK=100):
        self.beta = beta
        urm = self.URM_train * self.beta
        icm = self.ICM_train * (1 - self.beta)
        virtual_URM = sps.vstack([urm, icm.T])
        self.virtual_URM = virtual_URM.tocsr()
        
        self.__slim_rec = SLIMElasticNetRecommender(self.virtual_URM)
        
        self.__slim_rec.fit(l1_ratio=l1_ratio, alpha=alpha, positive_only=positive_only, topK=topK)
        self.W_sparse = self.__slim_rec.W_sparse
    """

    def fit(self, beta=0.5, l1_ratio=0.1, alpha = 1.0, positive_only=True, topK = 100):
        """
           beta is the parameter used to build the virtual URM that is given as input of the SLIM rec
        """
        assert l1_ratio>= 0 and l1_ratio<=1, "{}: l1_ratio must be between 0 and 1, provided value was {}".format(self.RECOMMENDER_NAME, l1_ratio)

        self.l1_ratio = l1_ratio
        self.positive_only = positive_only
        self.topK = topK

        #build the virtual URM
        self.beta = beta
        urm = self.URM_train * self.beta
        icm = self.ICM_train * (1 - self.beta)
        virtual_URM = sps.vstack([urm, icm.T])
        self.virtual_URM = virtual_URM.tocsr()

        # Display ConvergenceWarning only once and not for every item it occurs
        warnings.simplefilter("once", category = ConvergenceWarning)

        # initialize the ElasticNet model
        self.model = ElasticNet(alpha=alpha,
                                l1_ratio=self.l1_ratio,
                                positive=self.positive_only,
                                fit_intercept=False,
                                copy_X=False,
                                precompute=True,
                                selection='random',
                                max_iter=600,#here we have increased the max iter
                                tol=1e-4)

        virtual_URM = check_matrix(self.virtual_URM, 'csc', dtype=np.float32)

        n_items = virtual_URM.shape[1]

        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0

        start_time = time.time()
        start_time_printBatch = start_time

        # fit each item's factors sequentially (not in parallel)
        for currentItem in range(n_items):

            # get the target column
            y = virtual_URM[:, currentItem].toarray()

            # set the j-th column of X to zero
            start_pos = virtual_URM.indptr[currentItem]
            end_pos = virtual_URM.indptr[currentItem + 1]

            current_item_data_backup = virtual_URM.data[start_pos: end_pos].copy()
            virtual_URM.data[start_pos: end_pos] = 0.0

            # fit one ElasticNet model per column
            self.model.fit(virtual_URM, y)

            # self.model.coef_ contains the coefficient of the ElasticNet model
            # let's keep only the non-zero values

            # Select topK values
            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index

            nonzero_model_coef_index = self.model.sparse_coef_.indices
            nonzero_model_coef_value = self.model.sparse_coef_.data

            local_topK = min(len(nonzero_model_coef_value)-1, self.topK)

            relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topK)[0:local_topK]
            relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
            ranking = relevant_items_partition[relevant_items_partition_sorting]

            for index in range(len(ranking)):

                if numCells == len(rows):
                    rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                    cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                    values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))


                rows[numCells] = nonzero_model_coef_index[ranking[index]]
                cols[numCells] = currentItem
                values[numCells] = nonzero_model_coef_value[ranking[index]]

                numCells += 1

            # finally, replace the original values of the j-th column
            virtual_URM.data[start_pos:end_pos] = current_item_data_backup

            elapsed_time = time.time() - start_time
            new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)


            if time.time() - start_time_printBatch > 300 or currentItem == n_items-1:
                self._print("Processed {} ( {:.2f}% ) in {:.2f} {}. Items per second: {:.2f}".format(
                    currentItem+1,
                    100.0* float(currentItem+1)/n_items,
                    new_time_value,
                    new_time_unit,
                    float(currentItem)/elapsed_time))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        # generate the sparse weight matrix
        self.W_sparse = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                                       shape=(n_items, n_items), dtype=np.float32)

