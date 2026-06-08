import torch
import torch.nn as nn
from trainer import Trainer
from utils import use_cuda
import os


class Client(torch.nn.Module):
    def __init__(self, config):
        super(Client, self).__init__()
        self.config = config
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.embedding_user = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['client_model_layers'][:-1], config['client_model_layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=config['client_model_layers'][-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()
        
        if 'pretrain_path' in config and config['pretrain_path']:
            self.load_pretrain_weights(config['pretrain_path'], verbose=False)

    def forward(self, item_indices, pos_items=None):
        user_embedding = self.embedding_user(torch.tensor([0] * len(item_indices)).cpu())
        item_embedding = self.embedding_item(item_indices)
        vector = torch.cat([user_embedding, item_embedding], dim=-1)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.LeakyReLU()(vector)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass

    def load_pretrain_weights(self, pretrain_path, verbose=False):
        try:
            if os.path.exists(os.path.join(pretrain_path, 'item_embeddings.pth')):
                item_emb = torch.load(os.path.join(pretrain_path, 'item_embeddings.pth'))
                self.embedding_item.weight.data.copy_(item_emb)
            
            if os.path.exists(os.path.join(pretrain_path, 'user_embeddings.pth')):
                user_emb = torch.load(os.path.join(pretrain_path, 'user_embeddings.pth'))
                self.embedding_user.weight.data.copy_(user_emb)
            
            if os.path.exists(os.path.join(pretrain_path, 'network_params.pth')):
                network_params = torch.load(os.path.join(pretrain_path, 'network_params.pth'))
                
                for i in range(len(self.fc_layers)):
                    weight_key = f'fc_layers.{i}.weight'
                    bias_key = f'fc_layers.{i}.bias'
                    
                    if weight_key in network_params:
                        self.fc_layers[i].weight.data.copy_(network_params[weight_key])
                    if bias_key in network_params:
                        self.fc_layers[i].bias.data.copy_(network_params[bias_key])
                
                if 'affine_output.weight' in network_params:
                    self.affine_output.weight.data.copy_(network_params['affine_output.weight'])
                if 'affine_output.bias' in network_params:
                    self.affine_output.bias.data.copy_(network_params['affine_output.bias'])
                
            return True
        except Exception as e:
            return False


class Server(torch.nn.Module):
    def __init__(self, config):
        super(Server, self).__init__()
        self.config = config
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['client_model_layers'][:-1], config['client_model_layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=config['client_model_layers'][-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, item_indices):
        pass
    
    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        pass


class FedTrainer(Trainer):
    def __init__(self, config):
        self.client_model = Client(config)
        self.server_model = Server(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.client_model.cuda()
            self.server_model.cuda()
        self.mlp_keys = [k for k in self.client_model.state_dict().keys() if k.split('.')[0] in ['fc_layers', 'affine_output']]
        
        self.server_keys = self.mlp_keys
        
        if config['dataset'] == 'ali-ads':
            self.client_keys = ['embedding_user.weight']
        else:
            self.client_keys = ['embedding_user.weight', 'embedding_item.weight']
            
        self.use_individual_pretrain = config.get('use_individual_pretrain', False)
        self.pretrain_path = config.get('pretrain_path', '')
        
        super(FedTrainer, self).__init__(config)
        
    def load_client_pretrain_weights(self, model_client, user_id):
        if not self.pretrain_path:
            return model_client
            
        if self.use_individual_pretrain:
            client_pretrain_path = os.path.join(self.pretrain_path, f"client_{user_id}")
            if os.path.exists(client_pretrain_path):
                model_client.load_pretrain_weights(client_pretrain_path, verbose=False)
            
        return model_client
