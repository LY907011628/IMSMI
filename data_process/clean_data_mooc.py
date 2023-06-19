import pandas as pd


class Dataset(object):

    def __init__(self, source_file, data_file):
        self._source_file = source_file
        self._data_file = data_file

    def item_appear(self, appear_time, ds):
        dm = ds[['use_ID', 'ite_ID']].drop_duplicates()
        da = dm.groupby(by=['ite_ID'], as_index=False)['ite_ID'].agg({'cnt': 'count'})
        ite_list = list(da[da['cnt'] >= appear_time]['ite_ID'])
        ds = ds[ds['ite_ID'].isin(ite_list)]
        return ds

    def session_not_single(self, ds):
        ds = ds[ds.duplicated(['use_ID', 'time'], keep=False)]
        return ds

    def user_have_more_session(self, ds, user_sessin):
        dm = ds.drop_duplicates(['use_ID', 'time'])
        da = dm.groupby(by=['use_ID'], as_index=False)['use_ID'].agg({'cnt': 'count'})
        use_list = list(da[da['cnt'] >= user_sessin]['use_ID'])
        ds = ds[ds['use_ID'].isin(use_list)]
        return ds

    def first_step_clean_data(self, appear_time, user_sessin):
        data = pd.read_csv(self._source_file)
        ds = data.drop_duplicates()
        ds = ds.sort_values(by=['use_ID', 'time'])
        ds = self.item_appear(appear_time, ds)
        ds = self.user_have_more_session(ds, user_sessin)
        # ds = self.item_appear(appear_time, ds)
        ds.to_csv(self._data_file, index=False)


datatype = 'mooc'
source_file = '../data/' + datatype + '_data.csv'
data_file = '../data/' + datatype + '_clean_data.csv'
