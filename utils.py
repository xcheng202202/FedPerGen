import torch
import logging
import numpy as np
import pandas as pd
import random
import os
import datetime
import math


def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)


def use_optimizer(network, params):
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr=params['sgd_lr'],
                                    momentum=params['sgd_momentum'],
                                    weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), 
                                     lr=params['lr'],
                                     weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['rmsprop_lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer


def initLogging():
    path = 'log/'
    if not os.path.exists(path):
        os.makedirs(path)
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logFilename = os.path.join(path, 'ml-100k'+'.txt')

    logging.basicConfig(
                    level    = logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    datefmt  = '%y-%m-%d %H:%M',
                    filename = logFilename,
                    filemode = 'w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def load_data(config):
    dataset_dir = './' + config['dataset'] + '/u.data'
    if config['dataset'] == "ml-1m":
        rating = pd.read_csv(dataset_dir, sep=',', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
    elif config['dataset'] == "ml-100k":
        rating = pd.read_csv(dataset_dir, sep="\\t", header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
    elif config['dataset'] == "lastfm-2k":
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')
    elif config['dataset'] == "amazon":
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
        rating = rating.sort_values(by='uid', ascending=True)
    elif config['dataset'] == 'douban':
        rating = pd.read_csv(dataset_dir, sep=",", engine='python')
    elif config['dataset'] == 'bookcrossing':
        rating = pd.read_csv(dataset_dir, sep=",", engine='python')
    elif config['dataset'] == "ali-ads":
        rating = pd.read_csv(dataset_dir, sep=",", engine='python')
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")
    
    if config['dataset'] != "ali-ads":
        user_id = rating[['uid']].drop_duplicates().reindex()
        user_id['userId'] = np.arange(len(user_id))
        rating = pd.merge(rating, user_id, on=['uid'], how='left')
        item_id = rating[['mid']].drop_duplicates()
        item_id['itemId'] = np.arange(len(item_id))
        rating = pd.merge(rating, item_id, on=['mid'], how='left')
    else:
        rating['userId'] = rating['uid']
        rating['itemId'] = rating['mid']

    if config['dataset'] == 'douban' or config['dataset'] == 'bookcrossing':
        rating = rating[['userId', 'itemId', 'rating']]
    else:
        rating = rating[['userId', 'itemId', 'rating', 'timestamp']]
    logging.info('Range of userId is [{}, {}]'.format(rating.userId.min(), rating.userId.max()))
    logging.info('Range of itemId is [{}, {}]'.format(rating.itemId.min(), rating.itemId.max()))

    return rating


def compute_metrics(evaluate_data, test_scores, negative_scores, Ks):
    test_users, test_items = evaluate_data[0].cpu().data.view(-1).tolist(), evaluate_data[1].cpu().data.view(-1).tolist()
    neg_users, neg_items = evaluate_data[2].cpu().data.view(-1).tolist(), evaluate_data[3].cpu().data.view(-1).tolist()
    tst_scores, neg_scores = test_scores.data.view(-1).tolist(), negative_scores.data.view(-1).tolist()
    
    test = pd.DataFrame({'user': test_users,
                        'test_item': test_items,
                        'test_score': tst_scores})
    
    full = pd.DataFrame({'user': neg_users + test_users,
                        'item': neg_items + test_items,
                        'score': neg_scores + tst_scores})
    full = pd.merge(full, test, on=['user'], how='left')
    
    full['rank'] = full.groupby('user')['score'].rank(method='first', ascending=False)
    full.sort_values(['user', 'rank'], inplace=True)
    recall, precision, ndcg = [], [], []
    for at_k in Ks:
        top_k = full[full['rank']<=at_k]
        test_in_top_k = top_k[top_k['test_item'] == top_k['item']].copy()
        rec_k = len(test_in_top_k) * 1.0 / full['user'].nunique()
        test_in_top_k['ndcg'] = test_in_top_k['rank'].apply(lambda x: math.log(2) / math.log(1 + x))
        ndcg_k = test_in_top_k['ndcg'].sum() * 1.0 / full['user'].nunique()
        recall.append(rec_k)
        ndcg.append(ndcg_k)
        
    return recall, ndcg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_log_result(config, test_recalls, test_ndcgs, final_test_round):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    strr = current_time + '-' + 'Recall: ' + str(test_recalls[final_test_round]) + '-' \
        + '-' + 'NDCG: ' + str(test_ndcgs[final_test_round]) + '-' \
        + 'best_round: ' + str(final_test_round)
    sstrr = ''
    for k in config.keys():
        sstr = ', ' + f'{k}: ' + str(config[k])
        sstrr += sstr
    strr += sstrr
    file_name = "sh_result/"+config['dataset']+".txt"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'a') as file:
        file.write(strr + '\n')

    logging.info('recall_list: {}'.format(test_recalls))
    logging.info('ndcg_list: {}'.format(test_ndcgs))
    logging.info('config: {}'.format(sstrr))
    logging.info('Best test recall: {}, ndcg: {} at round {}'.format(test_recalls[final_test_round],
                                                                                    test_ndcgs[final_test_round],
                                                                                    final_test_round))
    logging.info('\n')
