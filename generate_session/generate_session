import pandas as pd
import numpy as np
import os


# generate sessions

class generate(object):

    def __init__(self, dataPath, sessPath):
        self._data = pd.read_csv(dataPath)
        self.sessPath = sessPath

    def stati_data(self):
        print(len(self._data))
        print(len(self._data.drop_duplicates(['use_ID', 'time'])))
        print(len(self._data) / len(self._data.drop_duplicates(['use_ID', 'time'])))
        print(len(self._data.drop_duplicates('use_ID')))
        print(len(self._data.drop_duplicates(['use_ID', 'time'])) / len(self._data.drop_duplicates('use_ID')))
        print(len(self._data.drop_duplicates('ite_ID')))

    def reform_u_i_id(self):
        user_to_id = {}
        item_to_id = {}
        user_count = 0
        item_count = 0
        for i in range(len(self._data)):
            u_id = self._data.at[i, 'use_ID']
            i_id = self._data.at[i, 'item_ID']
            if u_id in user_to_id.keys():
                self._data.at[i, 'use_ID'] = user_to_id[u_id]
            else:
                user_to_id[u_id] = user_count
                self._data.at[i, 'use_ID'] = user_count
                user_count += 1
            if i_id in item_to_id.keys():
                self._data.at[i, 'item_ID'] = item_to_id[i_id]
            else:
                item_to_id[i_id] = item_count
                self._data.at[i, 'item_ID'] = item_count
                item_count += 1
        self._data.to_csv('../data/middle_data.csv', index=False)
        print('user_count', user_count)
        print('item_count', item_count)

    def generate_session(self):
        self.stati_data()
        self.reform_u_i_id()  # recode
        self._data = pd.read_csv('../data/middle_data.csv')
        os.remove('../data/middle_data.csv')
        session_path = self.sessPath
        if os.path.exists(session_path):
            os.remove(session_path)
        session_file = open(session_path, 'a')

        user_num = len(self._data['use_ID'].drop_duplicates())
        item_num = len(self._data['item_ID'].drop_duplicates())
        session_file.write(str(user_num) + ',' + str(item_num) + '\n')
        last_userid = self._data.at[0, 'use_ID']
        last_time = self._data.at[0, 'time']
        session = str(last_userid) + ',' + str(self._data.at[0, 'item_ID'])
        for i in range(1, len(self._data)):
            userid = self._data.at[i, 'use_ID']
            itemid = self._data.at[i, 'item_ID']
            time = self._data.at[i, 'time']
            if userid == last_userid and time == last_time:
                session += ":" + str(itemid)
            elif userid != last_userid:
                session_file.write(session + '\n')
                last_userid = userid
                last_time = time
                session = str(userid) + ',' + str(itemid)
            else:
                session += '@' + str(itemid)
                last_time = time


# if __name__ == '__main__':
#     datatype = ['mooc_clean']
#     dataPath = '../data/' + datatype[0] + '_data.csv'
#     sessPath = '../data/' + datatype[0] + '_dataset.csv'
#     object = generate(dataPath, sessPath)
#     object.stati_data()
#     object.generate_session()
