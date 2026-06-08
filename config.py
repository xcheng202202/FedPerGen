import argparse

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def get_config():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--clients_sample_ratio', type=float, default=1.0)
    parser.add_argument('--clients_sample_num', type=int, default=0)
    parser.add_argument('--num_round', type=int, default=10)
    parser.add_argument('--local_epoch', type=int, default=1)
    parser.add_argument('--lr_eta', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--lr_client', type=float, default=0.5)
    parser.add_argument('--dataset', type=str, default='ml-100k')
    parser.add_argument('--num_users', type=int)
    parser.add_argument('--num_items', type=int)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--num_negative', type=int, default=5)
    parser.add_argument('--client_model_layers', type=str, default='128,32')
    parser.add_argument('--recall_k', type=str, default='5,10,15,20')
    parser.add_argument('--l2_regularization', type=float, default=0.)
    parser.add_argument('--use_cuda', type=boolean_string, default=False)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=99)
    parser.add_argument('--NUM_NEG', type=int, default=999)
    parser.add_argument('--earlystop', type=int, default=10)
    parser.add_argument('--use_diff_lr', type=boolean_string, default=True)

    parser.add_argument('--pretrain_path', type=str, default='./saved_clients')
    parser.add_argument('--use_individual_pretrain', type=boolean_string, default=True)
    parser.add_argument('--save_clients', type=boolean_string, default=True)
    parser.add_argument('--clients_save_path', type=str, default='./saved_clients/')

    parser.add_argument('--generator_hidden_size', type=int, default=128)
    parser.add_argument('--noise_dim', type=int, default=32)
    parser.add_argument('--generator_lr', type=float, default=0.001)
    parser.add_argument('--generator_weight_decay', type=float, default=1e-5)
    parser.add_argument('--n_pseudo_items', type=int, default=5)
    parser.add_argument('--diversity_weight', type=float, default=0.5)
    parser.add_argument('--contrast_weight', type=float, default=1.0)
    parser.add_argument('--alignment_weight', type=float, default=0.5)
    parser.add_argument('--pseudo_item_start_round', type=int, default=100)

    parser.add_argument('--train_ppmodel', type=boolean_string, default=False)
    parser.add_argument('--fed_mode', type=str, default='FedRec')
    parser.add_argument('--save_model', type=boolean_string, default=True)
    parser.add_argument('--save_name', type=str, default='FedGDA.pkl')

    parser.add_argument('--pri_epoch', type=int, default=350)
    parser.add_argument('--pri_batch', type=int, default=2000)
    parser.add_argument('--attack_mode', type=str, default='i_emb+mlp')
    parser.add_argument('--grad_based', type=boolean_string, default=False)
    parser.add_argument('--ITEM_NAME', type=str, default='updated_item')
    parser.add_argument('--PRI_TEST_RATIO', type=float, default=0.8)
    parser.add_argument('--GNN', type=boolean_string, default=False)
    parser.add_argument('--adam', type=boolean_string, default=False)
    parser.add_argument('--gnn_drop', type=float, default=0.5)
    parser.add_argument('--ep_min', type=float, default=30)
    parser.add_argument('--ep_max', type=float, default=60)

    args = parser.parse_args()

    config = vars(args)
    
    if len(config['recall_k']) > 1:
        config['recall_k'] = [int(item) for item in config['recall_k'].split(',')]
    else:
        config['recall_k'] = [int(config['recall_k'])]
    
    if len(config['client_model_layers']) > 1:
        config['client_model_layers'] = [int(item) for item in config['client_model_layers'].split(',')]
    else:
        config['client_model_layers'] = int(config['client_model_layers'])
    
    if config['dataset'] == 'ml-1m':
        config['num_users'] = 6040
        config['num_items'] = 3706
        config['NUM_NEG'] = 999
        config['gnn_drop'] = 0.1
        config['num_age'] = 3
        config['num_gender'] = 2
        config['num_occupation'] = 21
        config['pri_esti_round'] = 2
    elif config['dataset'] == 'ml-100k':
        config['num_users'] = 943
        config['num_items'] = 1682
        config['num_age'] = 3
        config['num_gender'] = 2
        config['num_occupation'] = 21
        config['pri_esti_round'] = 1
        config['NUM_NEG'] = 99
    elif config['dataset'] == 'ali-ads':
        config['num_users'] = 3198
        config['num_items'] = 4282
        config['num_age'] = 7
        config['num_gender'] = 2
        config['num_occupation'] = 2
        config['NUM_NEG'] = 999
        config['gnn_drop'] = 0.1
        config['pri_esti_round'] = 2
    
    if config['GNN']:
        config['batch_size'] = 512

    return config

if __name__ == '__main__':
    config = get_config()
    print(config)
