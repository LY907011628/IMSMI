class data_generation():
    def __init__(self, type):
        print('init')
        self.data_type = type
        print(self.data_type)
        self.dataset = './data/' + self.data_type + '_dataset.csv'
        self.train_users = []
        self.train_sessions = []
        self.train_items = []
        self.train_neg_items = []
        self.train_pre_sessions = []

        self.test_users = []
        self.test_candidate_items = []
        self.test_sessions = []
        self.test_pre_sessions = []
        self.test_real_items = []
        self.whole_test_real_itmes = []

        self.neg_number = 1
        self.user_number = 0
        self.item_number = 0
        self.train_batch_id = 0
        self.test_batch_id = 0
        self.records_number = 0

        self.train_users_t = []
        self.train_sessions_t = []
        self.train_items_t = []
        self.train_neg_items_t = []
        self.train_pre_sessions_t = []

        self.test_users_t = []
        self.test_candidate_items_t = []
        self.test_sessions_t = []
        self.test_pre_sessions_t = []

        self.neg_number_t = 1
        self.user_number_t = 0
        self.item_number_t = 0
        self.train_batch_id_t = 0
        self.test_batch_id_t = 0
        self.records_number_t = 0

    def gen_train_test_data(self):
        self.data = pd.read_csv(self.dataset, names=['user', 'sessions'], dtype='str')
        is_first_line = 1
        for line in self.data.values:
            if is_first_line:
                self.user_number = int(line[0])
                self.item_number = int(line[1])
                self.user_purchased_item = dict()
                is_first_line = 0
            else:
                user_id = int(line[0])
                sessions = [i for i in line[1].split('@')]
                size = len(sessions)
                the_first_session = [int(i) for i in sessions[0].split(':')]
                self.train_pre_sessions.append(the_first_session)
                tmp = copy.deepcopy(the_first_session)
                self.user_purchased_item[user_id] = tmp
                for j in range(1, size - 1):
                    self.train_users.append(user_id)
                    current_session = [int(it) for it in sessions[j].split(':')]
                    neg = self.gen_neg(user_id)
                    self.train_neg_items.append(neg)
                    if j != 1:
                        tmp = copy.deepcopy(self.user_purchased_item[user_id])
                        self.train_pre_sessions.append(tmp)
                    self.user_purchased_item[user_id].extend(current_session)
                    item = current_session[-1]
                    self.train_items.append(item)
                    current_session.remove(item)
                    self.train_sessions.append(current_session)
                    self.records_number += 1
                self.test_users.append(user_id)
                current_session = [int(it) for it in sessions[size - 1].split(':')]
                item = current_session[-1]
                self.test_real_items.append(int(item))
                current_session.remove(item)
                self.test_sessions.append(current_session)
                self.test_pre_sessions.append(self.user_purchased_item[user_id])
        print('test_real_item' + str(len(self.test_real_items)))
        self.test_candidate_items = list(range(self.item_number))
        print("self.test_candidate_items", self.test_candidate_items)

    def gen_neg(self, user_id):
        neg_item = np.random.randint(self.item_number)
        while neg_item in self.user_purchased_item[user_id]:
            neg_item = np.random.randint(self.item_number)
        return neg_item

    def gen_train_batch_data(self, batch_size):
        l = len(self.train_users)

        if self.train_batch_id == l:
            self.train_batch_id = 0

        batch_user = self.train_users[self.train_batch_id:self.train_batch_id + batch_size]
        batch_item = self.train_items[self.train_batch_id:self.train_batch_id + batch_size]
        batch_session = self.train_sessions[self.train_batch_id]
        batch_neg_item = self.train_neg_items[self.train_batch_id:self.train_batch_id + batch_size]
        batch_pre_session = self.train_pre_sessions[self.train_batch_id]

        self.train_batch_id = self.train_batch_id + batch_size

        return batch_user, batch_item, batch_session, batch_neg_item, batch_pre_session

    def gen_test_batch_data(self, user_id, batch_size):

        batch_user = self.test_users[user_id:user_id + batch_size]
        batch_item = self.test_candidate_items
        batch_session = self.test_sessions[user_id]
        batch_pre_session = self.test_pre_sessions[user_id]

        return batch_user, batch_item, batch_session, batch_pre_session


class data_generation_t():
    def __init__(self, type):
        print('init')
        self.data_type = type
        print(self.data_type)
        self.dataset_t = './data/mooc_month_dataset.csv'
        self.train_users_t = []
        self.train_sessions_t = []
        self.train_items_t = []
        self.train_neg_items_t = []
        self.train_pre_sessions_t = []

        self.test_users_t = []
        self.test_candidate_items_t = []
        self.test_sessions_t = []
        self.test_pre_sessions_t = []
        self.test_real_items_t = []
        self.whole_test_real_itmes_t = []

        self.neg_number_t = 1
        self.user_number_t = 0
        self.item_number_t = 0
        self.train_batch_id_t = 0
        self.test_batch_id_t = 0
        self.records_number_t = 0



    def gen_train_test_data_t(self):
        self.data_t = pd.read_csv(self.dataset_t, names=['user', 'sessions'], dtype='str')
        is_first_line = 1
        for line in self.data_t.values:
            if is_first_line:
                self.user_number_t = int(line[0])
                self.item_number_t = int(line[1])
                self.user_purchased_item_t = dict()
                is_first_line = 0
            else:
                user_id = int(line[0])
                sessions = [i for i in line[1].split('@')]
                size = len(sessions)
                the_first_session = [int(i) for i in sessions[0].split(':')]
                self.train_pre_sessions_t.append(the_first_session)
                tmp = copy.deepcopy(the_first_session)
                self.user_purchased_item_t[user_id] = tmp
                for j in range(1, size - 1):
                    self.train_users_t.append(user_id)
                    current_session = [int(it) for it in sessions[j].split(':')]
                    neg = self.gen_neg_t(user_id)
                    self.train_neg_items_t.append(neg)
                    if j != 1:
                        tmp = copy.deepcopy(self.user_purchased_item_t[user_id])
                        self.train_pre_sessions_t.append(tmp)
                    self.user_purchased_item_t[user_id].extend(current_session)
                    # item = random.choice(current_session)
                    item = current_session[-1]
                    self.train_items_t.append(item)
                    current_session.remove(item)
                    self.train_sessions_t.append(current_session)
                    self.records_number_t += 1
                self.test_users_t.append(user_id)
                current_session = [int(it) for it in sessions[size - 1].split(':')]
                # item = random.choice(current_session)
                item = current_session[-1]
                self.test_real_items_t.append(int(item))
                current_session.remove(item)
                self.test_sessions_t.append(current_session)
                self.test_pre_sessions_t.append(self.user_purchased_item_t[user_id])
        print('test_real_item' + str(len(self.test_real_items_t)))
        self.test_candidate_items_t = list(range(self.item_number_t))
        print("self.test_candidate_items", self.test_candidate_items_t)

    def gen_neg_t(self, user_id):
        neg_item = np.random.randint(self.item_number_t)
        while neg_item in self.user_purchased_item_t[user_id]:
            neg_item = np.random.randint(self.item_number_t)
        return neg_item

    def gen_train_batch_data_t(self, batch_size):
        l = len(self.train_users_t)

        if self.train_batch_id_t == l:
            self.train_batch_id_t = 0

        batch_user = self.train_users_t[self.train_batch_id_t:self.train_batch_id_t + batch_size]
        batch_item = self.train_items_t[self.train_batch_id_t:self.train_batch_id_t + batch_size]
        batch_session = self.train_sessions_t[self.train_batch_id_t]
        batch_neg_item = self.train_neg_items_t[self.train_batch_id_t:self.train_batch_id_t + batch_size]
        batch_pre_session = self.train_pre_sessions_t[self.train_batch_id_t]

        self.train_batch_id_t = self.train_batch_id_t + batch_size

        return batch_user, batch_item, batch_session, batch_neg_item, batch_pre_session

    def gen_test_batch_data_t(self, user_id, batch_size):

        batch_user = self.test_users_t[user_id:user_id + batch_size]
        batch_item = self.test_candidate_items_t
        batch_session = self.test_sessions_t[user_id]
        batch_pre_session = self.test_pre_sessions_t[user_id]
        # print("finish")
        return batch_user, batch_item, batch_session, batch_pre_session
