import pandas as pd
import numpy as np
import scipy.sparse as sp
import os

if __name__ == '__main__':
    print('RecSys challenge 2020 module to load the data')


class DataParser(object):
    def __init__(self, directory='data'):
        self.__path_dir = directory
        self.__train_fn = 'data_train.csv'
        self.__icm_fn = 'data_ICM_title_abstract.csv'
        self.__target_users_fn = 'data_target_users_test.csv'
        self.__load_data__()

    def __load_data__(self):
        self.__ratings_frame = pd.read_csv(os.path.join(os.getcwd(), self.__path_dir, self.__train_fn),
                                           header=0,
                                           names=['user_id', "item_id", "ratings"],
                                           dtype={'user_id': np.int32,
                                                  "item_id": np.int32,
                                                  "ratings": np.float64})
        self.__icm_frame = pd.read_csv(os.path.join(os.getcwd(), self.__path_dir, self.__icm_fn),
                                       header=0,
                                       names=['item_id', 'feature_id', 'value'],
                                       dtype={'item_id': np.int32,
                                              "feature_id": np.int32,
                                              "value": np.float64})

    def get_ratings(self):
        return self.__ratings_frame

    def get_icm_frame(self):
        return self.__icm_frame

    def get_mapped_ratings(self):
        unique_users = self.__ratings_frame.user_id.unique()
        unique_items = self.__ratings_frame.item_id.unique()

        num_users, min_user_id, max_user_id = unique_users.size, unique_users.min(), unique_users.max()
        num_items, min_item_id, max_item_id = unique_items.size, unique_items.min(), unique_items.max()

        mapping_user_id = pd.DataFrame({"mapped_user_id": np.arange(num_users), "user_id": unique_users})
        mapping_item_id = pd.DataFrame({"mapped_item_id": np.arange(num_items), "item_id": unique_items})

        mapped_frame = pd.merge(left=self.__ratings_frame,
                                right=mapping_user_id,
                                how="inner",
                                on="user_id")
        mapped_frame = pd.merge(left=mapped_frame,
                                right=mapping_item_id,
                                how="inner",
                                on="item_id")
        return mapped_frame
    
    def get_statistics(self):
        unique_users = self.__ratings_frame.user_id.unique()
        unique_items = self.__ratings_frame.item_id.unique()
        num_users, min_user_id, max_user_id = unique_users.size, unique_users.min(), unique_users.max()
        num_items, min_item_id, max_item_id = unique_items.size, unique_items.min(), unique_items.max()
        print(f"Users:\nnum_users={num_users}")
        print(f"min_user_id={min_user_id}")
        print(f"max_user_id={max_user_id}\n")
        print(f"Items:\nnum_items={num_items}")
        print(f"min_item_id={min_item_id}")
        print(f"max_item_id={max_item_id}\n")
        tot_ratings = self.__ratings_frame.size
        print(f"Ratings:\ntot_ratings={tot_ratings}")
        
        sparsity = float("{:.5f}".format(tot_ratings/(num_users * num_items)))
        print(f"sparsity={sparsity}")
        users_stats = {'num':num_users, 'min':min_user_id, 'max':max_user_id}
        items_stats = {'num':num_items, 'min':min_item_id, 'max':max_item_id}
        ratings_stats = {'num':tot_ratings, 'sparsity':sparsity}
        
        return users_stats, items_stats, ratings_stats

    def get_URM_all(self):
        unique_users = self.__ratings_frame.user_id.unique()
        unique_items = self.__ratings_frame.item_id.unique()
        max_user_id = unique_users.max()
        max_item_id = unique_items.max()
        urm_all = sp.csr_matrix((self.__ratings_frame.ratings, (self.__ratings_frame.user_id,
                                                                self.__ratings_frame.item_id)),
                                shape=(max_user_id+1, max_item_id+1))
        return urm_all

    def get_target_data(self):
        return pd.read_csv(os.path.join(os.getcwd(), self.__path_dir, self.__target_users_fn),
                           header=0,
                           names=["user_id"],
                           dtype=[('user_id', np.int32)])

    def get_ICM_all(self):
        num_features = max(self.__icm_frame.feature_id.to_list()) + 1
        num_items = max(self.__icm_frame.item_id.to_list()) + 1
        icm_shape = (num_items, num_features)
        icm_all = sp.csr_matrix((self.__icm_frame.value.to_list(), (self.__icm_frame.item_id.to_list(), self.__icm_frame.feature_id.to_list())), shape=icm_shape)
        return icm_all

    def filter_URM_test_by_range(self, URM_train, URM_test, filter_range=(0,-1)):
        """
        the method returns the URM_test filtered according to the number of interaction (specified) in the URM_train
        set max = -1 to specify an unbounded range
        """
        assert filter_range[0] >= 0 and (filter_range[1] > filter_range[0] or filter_range[1] == -1), "Invalid filter range."
        mat = URM_train.tocoo()
        d = {'user_id': mat.row,
             'item_id': mat.col,
             'rating': mat.data}
        frame_train = pd.DataFrame(data=d)
        frame_train = frame_train.groupby(['user_id']).size().reset_index(name='num_inter')

        mat = URM_test.tocoo()
        d = {'user_id': mat.row,
             'item_id': mat.col,
             'rating': mat.data}
        frame_test = pd.DataFrame(data=d)
        frame_test = frame_test.join(frame_train.set_index('user_id'), on='user_id')
        frame_test = frame_test.loc[frame_test['num_inter'] >= filter_range[0]]
        if not filter_range[1] == -1:
            frame_test = frame_test.loc[frame_test['num_inter'] < filter_range[1]]

        assert not frame_test.empty, "There are no user in the selected range."

        partial_urm_test = sp.csr_matrix((frame_test.rating, (frame_test.user_id, frame_test.item_id)),
                                         shape=URM_train.shape)

        nnz_new = len(np.where(partial_urm_test.getnnz(axis=1) > 0)[0])
        nnz_old = len(np.where(URM_test.getnnz(axis=1) > 0)[0])
        fraction = "{:.2f}".format( nnz_new/nnz_old)
        print(f"Warning: the URM_test filtered in {filter_range} has {nnz_new} of {nnz_old} total users in the original URM_test. ({fraction})")

        return partial_urm_test

    def get_Special_ICM_all(self):
        num_features = max(self.__icm_frame.feature_id.to_list()) + 1
        num_items = max(self.__icm_frame.item_id.to_list()) + 1
        icm_shape = (num_items, num_features)
        icm_all = sp.csr_matrix((self.__icm_frame.item_id.to_list(), (self.__icm_frame.item_id.to_list(), self.__icm_frame.feature_id.to_list())), shape=icm_shape)
        return icm_all
