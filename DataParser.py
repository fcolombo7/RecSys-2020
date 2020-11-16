import pandas as pd
import numpy as np
import os

if __name__ == '__main__':
    print('RecSys challenge 2020 module to load the data')


class DataParser(object):
    def __init__(self, directory='data'):
        self.__path_dir = directory
        self.__train_fn = 'data_train.csv'
        self.__icm_fn = 'data_icm_title_abstract.csv'
        self.__target_users_fn = 'data_train.csv'
        self.__load_data__()

    def __load_data__(self):
        self.__ratings_frame = pd.read_csv(os.path.join(self.__path_dir, self.__train_fn),
                                         header=0,
                                         names=['user_id', "item_id", "ratings"],
                                         dtype={'user_id': np.int32,
                                                "item_id": np.int32,
                                                "ratings": np.float32})
        self.__icm_frame = pd.read_csv(os.path.join(self.__path_dir, self.__train_fn),
                                     header=0,
                                     names=['item_id', 'feature_id', 'value'],
                                     dtype={'item_id': np.int32,
                                            "feature_id": np.int32,
                                            "value": np.float64})

    def get_ratings(self):
        return self.__ratings_frame

    def get_icm(self):
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
        print(f"Users:\n{num_users=}")
        print(f"{min_user_id=}")
        print(f"{max_user_id=}\n")
        print(f"Items:\n{num_items=}")
        print(f"{min_item_id=}")
        print(f"{max_item_id=}\n")
        tot_ratings = self.__ratings_frame.size
        print(f"Ratings:\n{tot_ratings=}")
        
        sparsity = float("{:.5f}".format(tot_ratings/(num_users * num_items)))
        print(f"{sparsity=}")
        users_stats = {'num':num_users, 'min':min_user_id, 'max':max_user_id}
        items_stats = {'num':num_items, 'min':min_item_id, 'max':max_item_id}
        ratings_stats = {'num':tot_ratings, 'sparsity':sparsity}
        
        return users_stats, items_stats, ratings_stats
        