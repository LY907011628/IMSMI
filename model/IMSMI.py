import numpy as np
import pandas as pd
import tensorflow as tf
import random
import copy
import logging
import logging.config
import os
import math
SEED = 2022
def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_random_seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
class IMSMI():
    def __init__(self, data_type):
        print('init ... ')
        self.input_data_type = data_type
        logging.config.fileConfig('logging.conf')
        self.logger = logging.getLogger()
        self.dg = data_generation(self.input_data_type)
        self.dg_t = data_generation_t(self.input_data_type)
        self.dg.gen_train_test_data()

        self.train_user_purchased_item_dict = self.dg.user_purchased_item

        self.user_number = self.dg.user_number
        self.item_number = self.dg.item_number
        self.neg_number = self.dg.neg_number

        self.test_users = self.dg.test_users
        self.test_candidate_items = self.dg.test_candidate_items
        self.test_sessions = self.dg.test_sessions
        self.test_pre_sessions = self.dg.test_pre_sessions
        self.test_real_items = self.dg.test_real_items

        self.global_dimension = 100
        self.batch_size = 1
        self.K = 3
        self.results = []
        self.best_result = 0
        self.step = 0
        self.lamada_u_v = 0.1
        self.lamada_a = 0.1

        self.initializer = tf.random_normal_initializer(mean=0, stddev=0.01)
        self.initializer_param = tf.random_uniform_initializer(minval=-np.sqrt(3 / self.global_dimension),
                                                               maxval=-np.sqrt(3 / self.global_dimension))

        self.user_id = tf.placeholder(tf.int32, shape=[None], name='user_id')
        self.item_id = tf.placeholder(tf.int32, shape=[None], name='item_id')
        self.current_session = tf.placeholder(tf.int32, shape=[None], name='current_session')
        self.pre_sessions = tf.placeholder(tf.int32, shape=[None], name='pre_sessions')
        self.neg_item_id = tf.placeholder(tf.int32, shape=[None], name='neg_item_id')

        self.user_embedding_matrix = tf.get_variable('user_embedding_matrix', initializer=self.initializer,
                                                     shape=[self.user_number, self.global_dimension])
        self.item_embedding_matrix = tf.get_variable('item_embedding_matrix', initializer=self.initializer,
                                                     shape=[self.item_number, self.global_dimension])
        self.the_first_w = tf.get_variable('the_first_w', initializer=self.initializer_param,
                                           shape=[self.global_dimension, self.global_dimension])
        self.the_second_w = tf.get_variable('the_second_w', initializer=self.initializer_param,
                                            shape=[self.global_dimension, self.global_dimension])
        self.the_first_bias = tf.get_variable('the_first_bias', initializer=self.initializer_param,
                                              shape=[self.global_dimension])
        self.the_second_bias = tf.get_variable('the_second_bias', initializer=self.initializer_param,
                                               shape=[self.global_dimension])
        self.the_third_w = tf.get_variable('the_third_w', initializer=self.initializer_param,
                                            shape=[self.global_dimension, self.global_dimension])
        self.the_third_bias = tf.get_variable('the_third_bias', initializer=self.initializer_param,
                                              shape=[self.global_dimension])#
        self.dg_t = data_generation_t(self.input_data_type)
        self.dg_t.gen_train_test_data_t()
        self.train_user_purchased_item_dict_t = self.dg_t.user_purchased_item_t

        self.user_number_t = self.dg_t.user_number_t
        self.item_number_t = self.dg_t.item_number_t
        self.neg_number_t = self.dg_t.neg_number_t

        self.test_users_t = self.dg_t.test_users_t
        self.test_candidate_items_t = self.dg_t.test_candidate_items_t
        self.test_sessions_t = self.dg_t.test_sessions_t
        self.test_pre_sessions_t = self.dg_t.test_pre_sessions_t
        self.test_real_items_t = self.dg_t.test_real_items_t

        self.global_dimension_t = self.global_dimension
        self.iteration = 100
        self.lamada_u_v_t = self.lamada_u_v
        self.lamada_a_t = self.lamada_a

        self.initializer = tf.random_normal_initializer(mean=0, stddev=0.01)
        self.initializer_param = tf.random_uniform_initializer(minval=-np.sqrt(3 / self.global_dimension_t),
                                                               maxval=-np.sqrt(3 / self.global_dimension_t))

        self.user_id_t = tf.placeholder(tf.int32, shape=[None], name='user_id_t')
        self.item_id_t = tf.placeholder(tf.int32, shape=[None], name='item_id_t')
        self.current_session_t = tf.placeholder(tf.int32, shape=[None], name='current_session_t')
        self.pre_sessions_t = tf.placeholder(tf.int32, shape=[None], name='pre_sessions_t')
        self.neg_item_id_t = tf.placeholder(tf.int32, shape=[None], name='neg_item_id_t')

        self.user_embedding_matrix_t = tf.get_variable('user_embedding_matrix_t', initializer=self.initializer,
                                                       shape=[self.user_number_t, self.global_dimension_t])
        self.item_embedding_matrix_t = tf.get_variable('item_embedding_matrix_t', initializer=self.initializer,
                                                       shape=[self.item_number_t, self.global_dimension_t])
        self.the_first_w_t = tf.get_variable('the_first_w_t', initializer=self.initializer_param,
                                             shape=[self.global_dimension_t, self.global_dimension_t])
        self.the_second_w_t = tf.get_variable('the_second_w_t', initializer=self.initializer_param,
                                              shape=[self.global_dimension_t, self.global_dimension_t])
        self.the_first_bias_t = tf.get_variable('the_first_bias_t', initializer=self.initializer_param,
                                                shape=[self.global_dimension_t])
        self.the_second_bias_t = tf.get_variable('the_second_bias_t', initializer=self.initializer_param,
                                                 shape=[self.global_dimension_t])
        self.the_third_w_t = tf.get_variable('the_third_w_t', initializer=self.initializer_param,
                                             shape=[self.global_dimension_t, self.global_dimension_t])
        self.the_third_bias_t = tf.get_variable('the_third_bias_t', initializer=self.initializer_param,
                                                shape=[self.global_dimension_t])


    def attention_level_one(self, user_embedding, pre_sessions_embedding, the_first_w, the_first_bias):
        self.weight =  tf.nn.softmax(tf.transpose(tf.matmul(tf.nn.relu(
            tf.add(tf.matmul(pre_sessions_embedding, the_first_w), the_first_bias)),  tf.transpose(user_embedding))))

        out = tf.reduce_sum(tf.multiply(pre_sessions_embedding, tf.transpose(self.weight)), axis=0)
        return out

    def attention_level_two(self, user_embedding, current_session_embedding, long_user_embedding, short_user_embedding, the_second_w, the_second_bias):
        current_long = tf.concat([current_session_embedding, tf.expand_dims(long_user_embedding, axis=0)], 0)
        self.weight = tf.nn.softmax(tf.transpose(tf.matmul(
            tf.nn.relu(tf.add(
                tf.matmul(
                    tf.concat([current_long, tf.expand_dims(short_user_embedding, axis=0)],0),
                    the_second_w),
                the_second_bias)), tf.transpose(user_embedding))))
        # '''
        self.weight = self.weight / math.sqrt(self.global_dimension_t)
        out = tf.reduce_sum(
            tf.multiply(tf.concat([current_long, tf.expand_dims(short_user_embedding, axis=0)], 0),
                        tf.transpose(self.weight)), axis=0)
        return out

    def attention_level_three(self, user_embedding, long_user_embedding, current_session_embedding, the_third_w, the_third_bias):
        self.weight = tf.nn.softmax(tf.transpose(tf.matmul(tf.nn.relu(
            tf.add(tf.matmul(current_session_embedding, the_third_w), the_third_bias)), tf.transpose(user_embedding))))
        out = tf.reduce_sum(tf.multiply(current_session_embedding, tf.transpose(self.weight)), axis=0)
        return out
    def attention_level_one_t(self, user_embedding, pre_sessions_embedding, the_first_w, the_first_bias):
        self.weight =  tf.nn.softmax(tf.transpose(tf.matmul(tf.nn.relu(
            tf.add(tf.matmul(pre_sessions_embedding, the_first_w), the_first_bias)),  tf.transpose(user_embedding))))

        out = tf.reduce_sum(tf.multiply(pre_sessions_embedding, tf.transpose(self.weight)), axis=0)
        return out

    def attention_level_two_t(self, user_embedding, current_session_embedding, long_user_embedding, short_user_embedding, the_second_w, the_second_bias):
        current_long = tf.concat([current_session_embedding, tf.expand_dims(long_user_embedding, axis=0)], 0)
        self.weight = tf.nn.softmax(tf.transpose(tf.matmul(
            tf.nn.relu(tf.add(
                tf.matmul(
                    tf.concat([current_long, tf.expand_dims(short_user_embedding, axis=0)],0),
                    the_second_w),
                the_second_bias)), tf.transpose(user_embedding))))
        # '''
        self.weight = self.weight / math.sqrt(self.global_dimension_t)
        out = tf.reduce_sum(
            tf.multiply(tf.concat([current_long, tf.expand_dims(short_user_embedding, axis=0)], 0),
                        tf.transpose(self.weight)), axis=0)
        return out

    def attention_level_three_t(self, user_embedding, long_user_embedding, current_session_embedding, the_third_w, the_third_bias):
        self.weight = tf.nn.softmax(tf.transpose(tf.matmul(tf.nn.relu(
            tf.add(tf.matmul(current_session_embedding, the_third_w), the_third_bias)), tf.transpose(user_embedding))))
        out = tf.reduce_sum(tf.multiply(current_session_embedding, tf.transpose(self.weight)), axis=0)
        return out

    def build_model(self):
        print('building model ... ')
        self.user_embedding = tf.nn.embedding_lookup(self.user_embedding_matrix, self.user_id)
        self.item_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.item_id)
        self.current_session_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.current_session)
        self.pre_sessions_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.pre_sessions)
        self.neg_item_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.neg_item_id)
        self.current_session_embedding_m = tf.nn.embedding_lookup(self.item_embedding_matrix, self.current_session_t)
        self.pre_sessions_embedding_m = tf.nn.embedding_lookup(self.item_embedding_matrix, self.pre_sessions_t)
        self.neg_item_embedding_m = tf.nn.embedding_lookup(self.item_embedding_matrix, self.neg_item_id_t)
        self.long_user_embedding = self.attention_level_one(self.user_embedding, self.pre_sessions_embedding,
                                                            self.the_first_w, self.the_first_bias)
        self.short_user_embedding = self.attention_level_three(self.user_embedding, self.long_user_embedding,
                                                               self.current_session_embedding,
                                                               self.the_third_w, self.the_third_bias)
        self.hybrid_user_embedding_d = self.attention_level_two(self.user_embedding, self.current_session_embedding,
                                                              self.long_user_embedding, self.short_user_embedding,
                                                              self.the_second_w, self.the_second_bias)
        self.long_user_embedding_m = self.attention_level_one_t(self.user_embedding, self.pre_sessions_embedding_m,
                                                            self.the_first_w_t, self.the_first_bias_t)
        self.short_user_embedding_m = self.attention_level_three_t(self.user_embedding, self.long_user_embedding_m,
                                                               self.current_session_embedding_m,
                                                               self.the_third_w_t, self.the_third_bias_t)
        self.hybrid_user_embedding_m = self.attention_level_two_t(self.user_embedding, self.current_session_embedding_m,
                                                              self.long_user_embedding_m, self.short_user_embedding_m,
                                                              self.the_second_w_t, self.the_second_bias_t)
        self.hybrid_user_embedding = self.hybrid_user_embedding_m + self.hybrid_user_embedding_d
        self.positive_element_wise = tf.matmul(tf.expand_dims(self.hybrid_user_embedding, axis=0),
                                               tf.transpose(self.item_embedding))
        self.negative_element_wise = tf.matmul(tf.expand_dims(self.hybrid_user_embedding, axis=0),
                                               tf.transpose(self.neg_item_embedding))
        self.intention_loss = tf.reduce_mean(
            -tf.log(tf.nn.sigmoid(self.positive_element_wise - self.negative_element_wise)))
        self.regular_loss_u_v = tf.add(self.lamada_u_v * tf.nn.l2_loss(self.user_embedding),
                                       self.lamada_u_v * tf.nn.l2_loss(self.item_embedding))

        self.regular_loss_a_d = tf.add(tf.add(self.lamada_a * tf.nn.l2_loss(self.the_first_w),
                                              self.lamada_a * tf.nn.l2_loss(self.the_third_w)),
                                       self.lamada_a * tf.nn.l2_loss(self.the_second_w)
                                       )
        self.regular_loss_a_t = tf.add(tf.add(self.lamada_a * tf.nn.l2_loss(self.the_first_w_t),
                                              self.lamada_a * tf.nn.l2_loss(self.the_third_w_t)),
                                       self.lamada_a * tf.nn.l2_loss(self.the_second_w_t)
                                       )
        self.regular_loss_a =tf.add(self.regular_loss_a_d, self.regular_loss_a_t)
        self.regular_loss = tf.add(self.regular_loss_a, self.regular_loss_u_v)
        self.intention_loss = tf.add(self.intention_loss, self.regular_loss)
        self.top_value, self.top_index = tf.nn.top_k(self.positive_element_wise, k=self.K, sorted=True)

    def run(self):
        print('running ... ')
        with tf.Session() as self.sess:
            self.intention_optimizer = tf.train.AdagradOptimizer(learning_rate=0.2).minimize(
                self.intention_loss)
            init = tf.global_variables_initializer()
            self.sess.run(init)
            for iter in range(self.iteration):
                print('new iteration begin ... ')
                print('iteration: ', str(iter))
                while self.step * self.batch_size < self.dg.records_number:
                    batch_user_t, batch_item_t, batch_session_t, batch_neg_item_t, batch_pre_sessions_t = self.dg_t.gen_train_batch_data_t(
                        self.batch_size)
                    batch_user, batch_item, batch_session, batch_neg_item, batch_pre_sessions = self.dg.gen_train_batch_data(
                        self.batch_size)
                    self.sess.run(self.intention_optimizer,
                                  feed_dict={self.user_id: batch_user,
                                             self.item_id: batch_item,
                                             self.current_session: batch_session,
                                             self.neg_item_id: batch_neg_item,
                                             self.pre_sessions: batch_pre_sessions,
                                             self.item_id_t: batch_item_t,
                                             self.current_session_t: batch_session_t,
                                             self.pre_sessions_t: batch_pre_sessions_t
                                             })
                    self.step += 1
                    if self.step * self.batch_size % 5000 == 0:
                        print('eval ...')
                        self.evolution(iter)
                self.step = 0

            # save
            self.save()

    def save(self):
        user_latent_factors, item_latent_factors, the_first_w, the_second_w, the_first_bias, the_second_bias, the_third_w, the_third_bias = self.sess.run(
            [self.user_embedding_matrix, self.item_embedding_matrix, self.the_first_w, self.the_second_w,
             self.the_first_bias, self.the_second_bias, self.the_third_w, self.the_third_bias])

        t = pd.DataFrame(user_latent_factors)
        t.to_csv('./mooc_result/user_latent_factors')

        t = pd.DataFrame(item_latent_factors)
        t.to_csv('./mooc_result/item_latent_factors')

        t = pd.DataFrame(the_first_w)
        t.to_csv('./mooc_result/the_first_w')

        t = pd.DataFrame(the_second_w)
        t.to_csv('./mooc_result/the_second_w')

        t = pd.DataFrame(the_first_bias)
        t.to_csv('./mooc_result/the_first_bias')

        t = pd.DataFrame(the_second_bias)
        t.to_csv('./mooc_result/the_second_bias')

        t = pd.DataFrame(the_third_w)
        t.to_csv('./mooc_result/the_third_w')

        t = pd.DataFrame(the_third_bias)
        t.to_csv('./mooc_result/the_third_bias')

        return


    def recall_k(self, pre_top_k, true_items):
        right_pre = 0
        user_number = len(pre_top_k)
        for i in range(user_number):
            if true_items[i] in pre_top_k[i]:
                right_pre += 1
        print("count", right_pre)
        return right_pre / user_number

    def MRR(self, pre_top_k, true_tiems):
        MRR_sum = 0
        use_number = len(pre_top_k)
        for i in range(use_number):
            if true_tiems[i] in pre_top_k[i]:
                for j in range(len(pre_top_k[i])):
                    if true_tiems[i] == pre_top_k[i][j]:
                        MRR_sum = MRR_sum + 1/(j+1)
        return MRR_sum/use_number

    def evolution(self, iter):
        pre_top_k = []
        for user_id in self.test_users:
            batch_user_t, batch_item_t, batch_session_t, batch_pre_session_t = self.dg_t.gen_test_batch_data_t(user_id,
                                                                                                       self.batch_size)
            batch_user, batch_item, batch_session, batch_pre_session = self.dg.gen_test_batch_data(user_id,
                                                                                                   self.batch_size)
            top_k_value, top_index = self.sess.run([self.top_value, self.top_index],
                                                   feed_dict={self.user_id: batch_user,
                                                              self.item_id: batch_item,
                                                              self.current_session: batch_session,
                                                              self.pre_sessions: batch_pre_session,
                                                              self.item_id_t: batch_item_t,
                                                              self.current_session_t: batch_session_t,
                                                              self.pre_sessions_t: batch_pre_session_t
                                                              })
            top_index_list = list(top_index[0])
            for i in range(len(self.train_user_purchased_item_dict[user_id])):
                if self.train_user_purchased_item_dict[user_id][i] in top_index_list:
                    top_index_list.remove(self.train_user_purchased_item_dict[user_id][i])
            pre_top_k.append(top_index_list[:self.K])  # top K
        precision_result = str(self.recall_k(pre_top_k, self.test_real_items, iter))
        MRR_result = str(self.MRR(pre_top_k, self.test_real_items, iter))
        self.logger.info('recall@' + str(self.K) + ' = ' + precision_result)
        if self.best_result < float(precision_result):
            self.best_result = float(precision_result)
            print("Best_result", self.best_result, "iter:", iter, "lamda a", self.lamada_a, "lanmda embedding", self.lamada_u_v)
            print("MRR_result", MRR_result)
        return


# if __name__ == '__main__':
#     # set_global_determinism(seed=SEED)
#     type = ['', 'mooc','mooc_clean']
#     model = IMSMI(type[2])
#     model.build_model()
#     model.run()
