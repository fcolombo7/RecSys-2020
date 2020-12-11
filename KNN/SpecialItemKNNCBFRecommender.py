from DataParser import DataParser
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender


class SpecialItemKNNCBFRecommender(ItemKNNCBFRecommender):
    RECOMMENDER_NAME = "Special-ItemKNNCBFRec"

    def __init__(self, URM_train, ICM_train):
        parser = DataParser()
        icm = parser.get_Special_ICM_all()
        super(SpecialItemKNNCBFRecommender, self).__init__(URM_train, icm)
        self.ICM_train=icm
