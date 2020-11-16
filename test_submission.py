import numpy as np
import scipy.sparse as sps
import pandas as pd
import re
from DataParser import DataParser
from CFItemKNN import CFItemKNN
from datetime import datetime


def create_csv(parser, recommender, name=None):

    out_userid = np.array([])
    out_itemlist = np.array([])

    target_data = parser.get_target_data()
    for user_id in target_data.user_id.unique():
        out_userid = np.append(out_userid, user_id)
        str_ = re.sub(' +', ' ', np.array_str(recommender.recommend(user_id, at=10)))[1:-1]
        if str_[0] == ' ':
            str_ = str_[1:]
        out_itemlist = np.append(out_itemlist, str_)

    out_dataframe = pd.DataFrame(data={'user_id':out_userid, 'item_list':out_itemlist})
    out_dataframe = out_dataframe.astype({'user_id': 'int32'})

    filename = str(datetime.now().strftime("res_%Y%m%d-%H%M.csv"))
    if not name is None:
        name = name + '_' + filename
    else:
        name = filename
    out_path = "res_csv/" + name
    out_dataframe.to_csv(out_path, index=False)

    # remove the single line added
    fd = open(out_path, "r")
    d = fd.read()
    fd.close()
    m = d.split("\n")
    s = "\n".join(m[:-1])
    fd = open(out_path, "w+")
    for i in range(len(s)):
        fd.write(s[i])
    fd.close()


if __name__ == '__main__':
    print("Making a submission... ")
    parser = DataParser()
    urm = parser.get_URM_all()
    # Define the recommender
    recommender = CFItemKNN(urm)
    recommender.fit(topK=967,shrink=356)
    create_csv(parser, recommender, 'testCFItem')
