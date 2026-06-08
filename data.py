import torch
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import Dataset
import os


class UserItemRatingDataset(Dataset):

    def __init__(self, user_tensor, item_tensor, target_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)
    
    
class SampleGenerator(object):

    def __init__(self, config, ratings):
        assert 'userId' in ratings.columns
        assert 'itemId' in ratings.columns
        assert 'rating' in ratings.columns

        self.NUM_NEG = config['NUM_NEG']
        self.ratings = ratings
        self.dataset = config['dataset']
        self.dataset_dir = "./" + config['dataset'] + "/" 
        self.num_negative = config['num_negative']
        self.preprocess_ratings = self._binarize(self.ratings)
        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())
        self.train_ratings, self.val_ratings, self.test_ratings = self._split_loo(self.preprocess_ratings)
        self.negatives = self._sample_negative(self.ratings)
    
    def get_data(self):
        if not os.path.exists(self.dataset_dir + "train.npy"):
            validate_data = self.validate_data
            test_data = self.test_data
            all_train_data = self.store_all_train_data(self.num_negative)
            torch.save(validate_data, self.dataset_dir+'val.npy')
            torch.save(test_data, self.dataset_dir+'test.npy')
            torch.save(all_train_data, self.dataset_dir+'train.npy')
        else:
            validate_data = torch.load(self.dataset_dir+'val.npy')
            test_data = torch.load(self.dataset_dir+'test.npy')
            all_train_data = torch.load(self.dataset_dir+'train.npy')
        return all_train_data, validate_data, test_data
        
    def _normalize(self, ratings):
        ratings = deepcopy(ratings)
        max_rating = ratings.rating.max()
        ratings['rating'] = ratings.rating * 1.0 / max_rating
        return ratings

    def _binarize(self, ratings):
        ratings = deepcopy(ratings)
        ratings['rating'][ratings['rating'] > 0] = 1.0
        return ratings

    def _split_loo(self, ratings):
        if self.dataset == 'douban' or self.dataset == 'bookcrossing':
            test = ratings.groupby('userId').tail(1)
            val = ratings.groupby('userId').nth(-2)
            train = ratings.drop(test.index).drop(val.index)
        else:
            ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
            test = ratings[ratings['rank_latest'] == 1]
            val = ratings[ratings['rank_latest'] == 2]
            train = ratings[ratings['rank_latest'] > 2]
        assert train['userId'].nunique() == test['userId'].nunique() == val['userId'].nunique()
        assert len(train) + len(test) + len(val) == len(ratings)
        return train[['userId', 'itemId', 'rating']], val[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

    def _sample_negative(self, ratings):
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        interact_status['val_negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(list(x), self.NUM_NEG))
        interact_status['tst_negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(list(x), self.NUM_NEG))

        train_interact_status = self.train_ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(columns={'itemId': 'interacted_items'})
        interact_status['negative_items'] = train_interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        return interact_status[['userId', 'negative_items', 'val_negative_samples', 'tst_negative_samples']]

    def store_all_train_data(self, num_negatives):
        users, items, ratings = [], [], []
        train_ratings = pd.merge(self.train_ratings, self.negatives[['userId', 'negative_items']], on='userId')
        train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(list(x), num_negatives))
        single_user = []
        user_item = []
        user_rating = []
        grouped_train_ratings = train_ratings.groupby('userId')
        train_users = []
        for userId, user_train_ratings in grouped_train_ratings:
            train_users.append(userId)
            user_length = len(user_train_ratings)
            for row in user_train_ratings.itertuples():
                single_user.append(int(row.userId))
                user_item.append(int(row.itemId))
                user_rating.append(float(row.rating))
                for i in range(num_negatives):
                    single_user.append(int(row.userId))
                    user_item.append(int(row.negatives[i]))
                    user_rating.append(float(0))
            assert len(single_user) == len(user_item) == len(user_rating)
            assert (1 + num_negatives) * user_length == len(single_user)
            users.append(single_user)
            items.append(user_item)
            ratings.append(user_rating)
            single_user = []
            user_item = []
            user_rating = []
        assert len(users) == len(items) == len(ratings) == len(self.user_pool)
        assert train_users == sorted(train_users)
        return [users, items, ratings]

    @property
    def validate_data(self):
        val_ratings = pd.merge(self.val_ratings, self.negatives[['userId', 'val_negative_samples']], on='userId')
        val_users, val_items, negative_users, negative_items = [], [], [], []
        for row in val_ratings.itertuples():
            val_users.append(int(row.userId))
            val_items.append(int(row.itemId))
            for i in range(int(len(row.val_negative_samples))):
                negative_users.append(int(row.userId))
                negative_items.append(int(row.val_negative_samples[i]))
        assert len(val_users) == len(val_items)
        assert len(negative_users) == len(negative_items)
        assert self.NUM_NEG * len(val_users) == len(negative_users)
        assert val_users == sorted(val_users)
        return [torch.LongTensor(val_users), torch.LongTensor(val_items), torch.LongTensor(negative_users),
                torch.LongTensor(negative_items)]

    @property
    def test_data(self):
        test_ratings = pd.merge(self.test_ratings, self.negatives[['userId', 'tst_negative_samples']], on='userId')
        test_ratings = test_ratings.sort_values('userId')
        test_users, test_items, negative_users, negative_items = [], [], [], []
        for row in test_ratings.itertuples():
            test_users.append(int(row.userId))
            test_items.append(int(row.itemId))
            for i in range(int(len(row.tst_negative_samples))):
                negative_users.append(int(row.userId))
                negative_items.append(int(row.tst_negative_samples[i]))
        assert len(test_users) == len(test_items)
        assert len(negative_users) == len(negative_items)
        assert self.NUM_NEG * len(test_users) == len(negative_users)
        assert test_users == sorted(test_users)
        return [torch.LongTensor(test_users), torch.LongTensor(test_items), torch.LongTensor(negative_users),
                torch.LongTensor(negative_items)]
