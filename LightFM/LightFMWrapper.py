from Base.BaseRecommender import BaseRecommender
from lightfm import LightFM
import numpy as np


class LightFMWrapper(BaseRecommender):
    """LightFMWrapper"""

    def save_model(self, folder_path, file_name=None):
        print('could not save the model.')
        pass

    RECOMMENDER_NAME = "LightFMWrapper"

    def __init__(self, URM_train, ICM_train):
        super(LightFMWrapper, self).__init__(URM_train)
        self.lightFM_model = None
        self.ICM_train = ICM_train.copy()

    def fit(self, alpha, num_components, num_epochs, learning_rate, NUM_THREADS=4):

        # Let's fit a WARP model
        self.lightFM_model = LightFM(loss='warp',
                                     item_alpha=alpha,
                                     learning_rate=learning_rate,
                                     no_components=num_components)

        self.lightFM_model = self.lightFM_model.fit(self.URM_train,
                                                    item_features=self.ICM_train,
                                                    epochs=num_epochs,
                                                    num_threads=NUM_THREADS)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        # Create a single (n_items, ) array with the item score, then copy it for every user
        if items_to_compute is None:
            items_to_compute = np.arange(self.n_items)
            item_features = self.ICM_train
        else:
            items_to_compute = np.array(items_to_compute)
            item_features = self.ICM_train[items_to_compute, :]

        item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf

        for user_index, user_id in enumerate(user_id_array):
            item_scores[user_index] = self.lightFM_model.predict(int(user_id),
                                                                 items_to_compute,
                                                                 item_features=item_features)
        return item_scores
