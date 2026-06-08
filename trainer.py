import torch
from utils import *
import numpy as np
import copy
import os
from data import UserItemRatingDataset
from torch.utils.data import DataLoader
from pseudo_item_embedding import PseudoItemGenerator, generate_diverse_noises, train_pseudo_item_generator

FIXED_WARMUP_ROUNDS = 80


class Trainer(object):

    def __init__(self, config):
        self.config = config
        self.server_model_param = {}
        self.client_model_params = {}
        self.client_crit = torch.nn.BCELoss()
        self.historical_noises = []
        self.original_num_items = config['num_items']
        self.permanent_client_item_embeddings = {}
        self.client_embeddings_frozen = False
        self.init_pseudo_item_generator(config)

    def init_pseudo_item_generator(self, config):
        embed_size = config['latent_dim']
        self.pseudo_item_generator = PseudoItemGenerator(
            embed_size=embed_size,
            hidden_size=config.get('generator_hidden_size', 128),
            noise_dim=config.get('noise_dim', 32)
        )

        if config['use_cuda'] is True:
            self.pseudo_item_generator.cuda()

        self.generator_optimizer = torch.optim.Adam(
            self.pseudo_item_generator.parameters(),
            lr=config.get('generator_lr', 0.001),
            weight_decay=config.get('generator_weight_decay', 1e-5)
        )

        self.diversity_weight = 0.5
        self.contrast_weight = 1.0
        self.alignment_weight = 0.5
        self.n_pseudo_items = config.get('n_pseudo_items', 5)

    def instance_user_train_loader(self, user_train_data):
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(user_train_data[0]),
                                        item_tensor=torch.LongTensor(user_train_data[1]),
                                        target_tensor=torch.FloatTensor(user_train_data[2]))
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

    def fed_train_single_batch(self, model_client, batch_data, optimizers, pos_items, round_id):
        _, items, ratings = batch_data[0], batch_data[1], batch_data[2]
        ratings = ratings.float()

        if self.config['use_cuda'] is True:
            items, ratings = items.cuda(), ratings.cuda()

        optimizer, optimizer_u, optimizer_i = optimizers
        optimizer.zero_grad()
        optimizer_u.zero_grad()
        optimizer_i.zero_grad()
        ratings_pred = model_client(items, pos_items)
        loss = self.client_crit(ratings_pred.view(-1), ratings)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_client.parameters(), 5)
        optimizer.step()
        optimizer_u.step()
        optimizer_i.step()
        return model_client, loss.item()

    def freeze_client_embeddings(self):
        self.client_embeddings_frozen = True
        logging.info("Client embeddings frozen")

    def aggregate_item_embeddings(self):
        if 'embedding_item.weight' not in self.server_model_param:
            first_client_with_emb = None
            for user_id in self.client_model_params:
                if 'embedding_item.weight' in self.client_model_params[user_id]:
                    first_client_with_emb = user_id
                    break
            
            if first_client_with_emb is not None:
                self.server_model_param['embedding_item.weight'] = copy.deepcopy(
                    self.client_model_params[first_client_with_emb]['embedding_item.weight']
                )

    def aggregate_clients_params_peruser(self, client_param, init_param, num_part, t, round_id, user_id):
        if t == 0:
            self.tmp_server_model_param = {}
            for key in self.server_keys:
                self.tmp_server_model_param[key] = copy.deepcopy(client_param[key].data).cpu()

            if round_id == 0 and not self.client_embeddings_frozen:
                if user_id not in self.permanent_client_item_embeddings:
                    if 'embedding_item.weight' in client_param and 'embedding_item.weight' in init_param:
                        item_emb_before = init_param['embedding_item.weight'].data.cpu()
                        item_emb_after = client_param['embedding_item.weight'].data.cpu()

                        real_item_emb_before = item_emb_before[:self.original_num_items]
                        real_item_emb_after = item_emb_after[:self.original_num_items]

                        changed_rows = torch.abs(real_item_emb_after - real_item_emb_before).sum(dim=1) > 1e-6

                        if torch.any(changed_rows):
                            changed_embeddings = real_item_emb_after[changed_rows]
                            avg_embedding = torch.mean(changed_embeddings, dim=0, keepdim=True)

                            self.permanent_client_item_embeddings[user_id] = {
                                'avg_embedding': avg_embedding,
                                'computed_round': 0
                            }

        else:
            for key in self.server_keys:
                user_params = copy.deepcopy(client_param[key].data).cpu()
                self.tmp_server_model_param[key].data += user_params

            if round_id == 0 and not self.client_embeddings_frozen:
                if user_id not in self.permanent_client_item_embeddings:
                    if 'embedding_item.weight' in client_param and 'embedding_item.weight' in init_param:
                        item_emb_before = init_param['embedding_item.weight'].data.cpu()
                        item_emb_after = client_param['embedding_item.weight'].data.cpu()

                        real_item_emb_before = item_emb_before[:self.original_num_items]
                        real_item_emb_after = item_emb_after[:self.original_num_items]

                        changed_rows = torch.abs(real_item_emb_after - real_item_emb_before).sum(dim=1) > 1e-6

                        if torch.any(changed_rows):
                            changed_embeddings = real_item_emb_after[changed_rows]
                            avg_embedding = torch.mean(changed_embeddings, dim=0, keepdim=True)

                            self.permanent_client_item_embeddings[user_id] = {
                                'avg_embedding': avg_embedding,
                                'computed_round': 0
                            }

        if t == num_part - 1:
            self.server_model_param = {}
            for key in self.server_keys:
                self.server_model_param[key] = self.tmp_server_model_param[key].data / num_part

            self.server_model_param['client_item_embeddings'] = self.permanent_client_item_embeddings

    def fed_train_a_round(self, all_train_data, round_id, warmup_phase_ended=False):
        if self.config['clients_sample_ratio'] <= 1:
            num_participants = int(self.config['num_users'] * self.config['clients_sample_ratio'])
            participants = np.random.choice(self.config['num_users'], num_participants, replace=False)
        else:
            participants = np.random.choice(self.config['num_users'], self.config['clients_sample_num'], replace=False)

        all_loss = 0
        for uidx, user in enumerate(participants):
            model_client = copy.deepcopy(self.client_model)

            if round_id == 0 and hasattr(self, 'load_client_pretrain_weights'):
                model_client = self.load_client_pretrain_weights(model_client, user)

            if round_id != 0:
                user_param_dict = self.load_local_param(user=user)
                model_client.load_state_dict(user_param_dict)

            init_param = copy.deepcopy(model_client.state_dict())

            if warmup_phase_ended:
                current_lr = 0.05
            else:
                current_lr = self.config['lr_client']

            optimizer = torch.optim.SGD(
                [{"params": model_client.fc_layers.parameters()},
                 {"params": model_client.affine_output.parameters()}],
                lr=current_lr)

            uemb_lr = current_lr / self.config['clients_sample_ratio'] * self.config['lr_eta'] - current_lr
            iemb_lr = self.config['lr_client'] * self.config['num_items'] * self.config['lr_eta'] - self.config['lr_client']

            optimizer_u = torch.optim.SGD(model_client.embedding_user.parameters(), lr=uemb_lr)
            optimizer_i = torch.optim.SGD(model_client.embedding_item.parameters(), lr=iemb_lr)

            optimizers = [optimizer, optimizer_u, optimizer_i]

            user_train_data = [all_train_data[0][user], all_train_data[1][user], all_train_data[2][user]]

            if round_id > 0 and hasattr(self, 'pseudo_items_dict') and user in self.pseudo_items_dict:
                if 'embedding_item.weight' in self.client_model_params[user]:
                    current_item_emb_size = self.client_model_params[user]['embedding_item.weight'].shape[0]

                    if current_item_emb_size > self.original_num_items and round_id >= self.config['pseudo_item_start_round']:
                        new_embedding_item = torch.nn.Embedding(
                            num_embeddings=current_item_emb_size,
                            embedding_dim=self.config['latent_dim']
                        )

                        with torch.no_grad():
                            if self.config['use_cuda']:
                                new_embedding_item = new_embedding_item.cuda()
                                existing_weights = self.client_model_params[user]['embedding_item.weight'].cuda()
                            else:
                                existing_weights = self.client_model_params[user]['embedding_item.weight']

                            new_embedding_item.weight.data[:existing_weights.size(0)] = existing_weights

                        model_client.embedding_item = new_embedding_item

                        if not hasattr(self, 'pseudo_indices_dict'):
                            self.pseudo_indices_dict = {}

                        if user in self.pseudo_indices_dict:
                            all_pseudo_indices = self.pseudo_indices_dict[user]
                        else:
                            all_pseudo_indices = list(range(self.original_num_items, current_item_emb_size))

                        train_users = user_train_data[0]
                        train_items = user_train_data[1]
                        train_ratings = user_train_data[2]

                        existing_items = set(train_items)

                        for pseudo_idx in all_pseudo_indices:
                            if pseudo_idx not in existing_items:
                                train_users.append(train_users[0])
                                train_items.append(pseudo_idx)
                                train_ratings.append(1.0)

                        user_train_data = [train_users, train_items, train_ratings]

            user_dataloader = self.instance_user_train_loader(user_train_data)
            model_client.train()

            epoch_loss = 0
            sample_num = 0
            ratings = torch.FloatTensor(user_train_data[2])
            items = torch.LongTensor(user_train_data[1])
            pos_mask = ratings > 1e-5
            pos_items = items[pos_mask]
            for epoch in range(self.config['local_epoch']):
                for batch_id, batch in enumerate(user_dataloader):
                    assert isinstance(batch[0], torch.LongTensor)
                    model_client, loss_batch = self.fed_train_single_batch(model_client, batch, optimizers, pos_items, round_id)
                    epoch_loss += loss_batch * len(batch[0])
                    sample_num += len(batch[0])

            all_loss += epoch_loss / sample_num

            client_param = model_client.state_dict()

            self.client_model_params[user] = {}
            for key in self.client_keys:
                self.client_model_params[user][key] = copy.deepcopy(client_param[key].data).cpu()

            avg_item_emb = torch.mean(client_param['embedding_item.weight'].data[items][pos_mask], dim=0)
            self.client_model_params[user][self.config['ITEM_NAME']] = copy.deepcopy(avg_item_emb).cpu()
            self.client_model_params[user]['pos_items'] = copy.deepcopy(pos_items).cpu()

            network_params = {}
            for key in client_param.keys():
                if key.startswith('fc_layers') or key.startswith('affine_output'):
                    network_params[key] = copy.deepcopy(client_param[key].data).cpu()

            if 'network_params_for_alignment' not in self.client_model_params[user]:
                self.client_model_params[user]['network_params_for_alignment'] = network_params

            self.aggregate_clients_params_peruser(client_param, init_param, len(participants), uidx, round_id, user)

        return all_loss / len(participants)

    def load_local_param(self, user):
        user_param_dict = copy.deepcopy(self.client_model.state_dict())

        for key in self.server_keys:
            if key in self.server_model_param.keys():
                user_param_dict[key] = copy.deepcopy(self.server_model_param[key].data).cpu()

        if user in self.client_model_params.keys():
            for key in self.client_keys:
                if key in self.client_model_params[user].keys() and self.client_model_params[user][key].shape == user_param_dict[key].shape:
                    user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).cpu()

        if 'embedding_item.weight' in self.server_model_param:
            user_param_dict['embedding_item.weight'] = copy.deepcopy(self.server_model_param['embedding_item.weight'].data).cpu()

        return user_param_dict

    def load_local_param_real_items(self, user):
        user_param_dict = copy.deepcopy(self.client_model.state_dict())

        for key in self.server_keys:
            if key in self.server_model_param.keys():
                user_param_dict[key] = copy.deepcopy(self.server_model_param[key].data).cpu()

        if user in self.client_model_params.keys():
            if 'embedding_user.weight' in self.client_model_params[user]:
                user_param_dict['embedding_user.weight'] = copy.deepcopy(self.client_model_params[user]['embedding_user.weight'].data).cpu()

            if 'embedding_item.weight' in self.client_model_params[user]:
                item_emb = copy.deepcopy(self.client_model_params[user]['embedding_item.weight'].data).cpu()
                if item_emb.shape[0] > self.original_num_items:
                    item_emb = item_emb[:self.original_num_items]
                user_param_dict['embedding_item.weight'] = item_emb

        return user_param_dict

    def fed_evaluate(self, evaluate_data):
        y = torch.FloatTensor([1] + [0] * self.config['NUM_NEG'])
        _, test_items = evaluate_data[0], evaluate_data[1]
        _, negative_items = evaluate_data[2], evaluate_data[3]
        if self.config['use_cuda'] is True:
            test_items = test_items.cuda()
            negative_items = negative_items.cuda()
            y = y.cuda()

        test_scores, negative_scores = None, None
        all_loss = 0
        for user in range(self.config['num_users']):
            user_model = copy.deepcopy(self.client_model)
            user_param_dict = self.load_local_param_real_items(user)
            user_model.load_state_dict(user_param_dict)
            user_model.eval()

            with torch.no_grad():
                test_item = test_items[user: user + 1]
                negative_item = negative_items[user * self.config['NUM_NEG']: (user + 1) * self.config['NUM_NEG']]
                pos_items = self.client_model_params[user]['pos_items']
                test_score = user_model(test_item, pos_items)
                negative_score = user_model(negative_item, pos_items)
                y_hat = torch.cat((test_score, negative_score))
                loss = self.client_crit(y_hat.view(-1), y)
                if user == 0:
                    test_scores = test_score
                    negative_scores = negative_score
                else:
                    test_scores = torch.cat((test_scores, test_score))
                    negative_scores = torch.cat((negative_scores, negative_score))
            all_loss += loss.item()

        test_scores, negative_scores = test_scores.cpu(), negative_scores.cpu()
        recall, ndcg = compute_metrics(evaluate_data, test_scores, negative_scores, self.config['recall_k'])
        return recall, ndcg, all_loss / self.config['num_users']

    def get_params(self):
        save_params = {
            'server': copy.deepcopy(self.server_model_param),
            'client': copy.deepcopy(self.client_model_params)
        }
        return save_params

    def save_clients_params(self, save_dir=None):
        if save_dir is None:
            save_dir = self.config.get('clients_save_path', './saved_clients/')

        os.makedirs(save_dir, exist_ok=True)

        for user_id in range(self.config['num_users']):
            client_dir = os.path.join(save_dir, f"client_{user_id}")
            os.makedirs(client_dir, exist_ok=True)

            if user_id in self.client_model_params:
                client_info = {
                    'user_id': user_id,
                    'pos_items': self.client_model_params[user_id].get('pos_items', None)
                }
                torch.save(client_info, os.path.join(client_dir, 'client_info.pth'))

                if 'embedding_user.weight' in self.client_model_params[user_id]:
                    torch.save(
                        self.client_model_params[user_id]['embedding_user.weight'],
                        os.path.join(client_dir, 'user_embeddings.pth')
                    )

                if self.config['ITEM_NAME'] in self.client_model_params[user_id]:
                    torch.save(
                        self.client_model_params[user_id][self.config['ITEM_NAME']],
                        os.path.join(client_dir, 'item_embeddings.pth')
                    )

            torch.save(
                {'config': self.config},
                os.path.join(client_dir, 'global_info.pth')
            )

            network_params = {}
            for key in self.server_model_param:
                if key.split('.')[0] in ['fc_layers', 'affine_output']:
                    network_params[key] = self.server_model_param[key]

            torch.save(network_params, os.path.join(client_dir, 'network_params.pth'))

            if 'embedding_item.weight' in self.server_model_param:
                torch.save(
                    self.server_model_param['embedding_item.weight'],
                    os.path.join(client_dir, 'item_embeddings.pth')
                )

    def _add_collected_pseudo_items_to_clients(self, collected_pseudo_items, diverse_noises):
        if not hasattr(self, 'pseudo_items_dict'):
            self.pseudo_items_dict = {}
        if not hasattr(self, 'pseudo_indices_dict'):
            self.pseudo_indices_dict = {}

        for client_idx, pseudo_items_list in collected_pseudo_items.items():
            if client_idx not in self.pseudo_items_dict:
                self.pseudo_items_dict[client_idx] = []

            for pseudo_item in pseudo_items_list:
                self.pseudo_items_dict[client_idx].append(pseudo_item)

            if client_idx in self.client_model_params:
                if 'embedding_item.weight' in self.client_model_params[client_idx]:
                    current_emb = self.client_model_params[client_idx]['embedding_item.weight']
                else:
                    current_emb = self.server_model_param.get('embedding_item.weight', None)

                if current_emb is not None:
                    current_size = current_emb.size(0)
                    new_pseudo_tensor = torch.cat([p if p.dim() == 2 else p.unsqueeze(0) for p in pseudo_items_list], dim=0)
                    new_emb = torch.cat([current_emb, new_pseudo_tensor], dim=0)
                    self.client_model_params[client_idx]['embedding_item.weight'] = new_emb

                    if client_idx not in self.pseudo_indices_dict:
                        self.pseudo_indices_dict[client_idx] = []
                    for i in range(len(pseudo_items_list)):
                        self.pseudo_indices_dict[client_idx].append(current_size + i)

    def run_experiment(self, config, sample_generator):
        train_losses = []
        test_losses = []
        test_recalls, test_ndcgs, test_hrs = [], [], []
        generator_losses = {
            'total': [],
            'diversity': [],
            'contrast': [],
            'alignment': [],
            'fc1_grad_norm': [],
            'fc2_grad_norm': [],
            'fc1_weight_norm': [],
            'fc2_weight_norm': []
        }

        best_recall, final_test_round = 0, 0
        test_data = sample_generator.test_data

        save_dir = 'pupe'
        os.makedirs(save_dir, exist_ok=True)

        patience_rounds = FIXED_WARMUP_ROUNDS if FIXED_WARMUP_ROUNDS is not None else 40
        best_fc1_weight_norm = -float('inf')
        rounds_without_improvement = 0
        warmup_phase_ended = False

        for round in range(3000):
            logging.info('-' * 80)
            logging.info('Round {} starts !'.format(round))

            if round == 0:
                all_train_data = sample_generator.store_all_train_data(config['num_negative'])
                train_loss = self.fed_train_a_round(all_train_data, round, warmup_phase_ended=False)
                train_losses.append(train_loss)
                logging.info('Trn_Loss={:.5f}'.format(train_loss))

                self.freeze_client_embeddings()
                self.aggregate_item_embeddings()

                if 'embedding_item.weight' in self.server_model_param:
                    server_original_emb = self.server_model_param['embedding_item.weight']
                    for user in self.client_model_params:
                        if 'embedding_item.weight' in self.client_model_params[user]:
                            self.client_model_params[user]['embedding_item.weight'] = server_original_emb.clone()

                self.round_1_client_params = copy.deepcopy(self.client_model_params)
                self.round_1_server_params = copy.deepcopy(self.server_model_param)

            elif not warmup_phase_ended:
                train_losses.append(train_losses[0] if train_losses else 0.0)
                self.client_model_params = copy.deepcopy(self.round_1_client_params)
                self.server_model_param = copy.deepcopy(self.round_1_server_params)

            else:
                all_train_data = sample_generator.store_all_train_data(config['num_negative'])
                train_loss = self.fed_train_a_round(all_train_data, round, warmup_phase_ended=True)
                train_losses.append(train_loss)
                logging.info('Trn_Loss={:.5f}'.format(train_loss))

                self.aggregate_item_embeddings()

                if 'embedding_item.weight' in self.server_model_param:
                    server_original_emb = self.server_model_param['embedding_item.weight']
                    for user in self.client_model_params:
                        if 'embedding_item.weight' in self.client_model_params[user]:
                            client_full_emb = self.client_model_params[user]['embedding_item.weight']
                            if client_full_emb.size(0) > self.original_num_items:
                                client_full_emb[:self.original_num_items, :] = server_original_emb
                            else:
                                self.client_model_params[user]['embedding_item.weight'] = server_original_emb.clone()

            noise_dim = self.pseudo_item_generator.noise_dim
            device = 'cuda' if self.config['use_cuda'] else 'cpu'
            diverse_noises, self.historical_noises = generate_diverse_noises(
                n_samples=5,
                noise_dim=noise_dim,
                device=device,
                min_distance=7,
                historical_noises=self.historical_noises
            )

            valid_client_count = len(self.permanent_client_item_embeddings)
            if valid_client_count > 0:
                self.server_model_param['client_item_embeddings'] = self.permanent_client_item_embeddings

                if not warmup_phase_ended:
                    generator_epochs = 3
                else:
                    generator_epochs = 2

                updated_generator, avg_losses, collected_pseudo_items = train_pseudo_item_generator(
                    generator=self.pseudo_item_generator,
                    optimizer=self.generator_optimizer,
                    client_models=self.client_model_params,
                    server_model_params=self.server_model_param,
                    diverse_noises=diverse_noises,
                    n_epochs=generator_epochs,
                    lambda1=self.alignment_weight,
                    lambda2=self.diversity_weight,
                    tau=0.1,
                    device=device,
                    verbose=False,
                    batch_size=128
                )

                self.pseudo_item_generator = updated_generator

                fc1_weight_norm = self.pseudo_item_generator.fc1.weight.norm().item()
                fc2_weight_norm = self.pseudo_item_generator.fc2.weight.norm().item()

                if not warmup_phase_ended:
                    if FIXED_WARMUP_ROUNDS is not None:
                        if round >= FIXED_WARMUP_ROUNDS - 1:
                            warmup_phase_ended = True
                    else:
                        if fc1_weight_norm > best_fc1_weight_norm:
                            best_fc1_weight_norm = fc1_weight_norm
                            rounds_without_improvement = 0
                        else:
                            rounds_without_improvement += 1

                        if rounds_without_improvement >= patience_rounds:
                            warmup_phase_ended = True

                generator_losses['total'].append(avg_losses.get('total', 0))
                generator_losses['diversity'].append(avg_losses.get('diversity', 0))
                generator_losses['contrast'].append(avg_losses.get('contrast', 0))
                generator_losses['alignment'].append(avg_losses.get('alignment', 0))
                generator_losses['fc1_grad_norm'].append(avg_losses.get('fc1_grad_norm', 0))
                generator_losses['fc2_grad_norm'].append(avg_losses.get('fc2_grad_norm', 0))
                generator_losses['fc1_weight_norm'].append(fc1_weight_norm)
                generator_losses['fc2_weight_norm'].append(fc2_weight_norm)

                if not warmup_phase_ended:
                    del collected_pseudo_items
                else:
                    self._add_collected_pseudo_items_to_clients(collected_pseudo_items, diverse_noises)

            else:
                for key in generator_losses.keys():
                    generator_losses[key].append(0.0)

            test_recall, test_ndcg, test_loss = self.fed_evaluate(test_data)
            test_hr = test_recall

            test_losses.append(test_loss)
            test_recalls.append(test_recall)
            test_ndcgs.append(test_ndcg)
            test_hrs.append(test_hr)

            logging.info(result2str_extended('Recall', config['recall_k'], test_recall))
            logging.info(result2str_extended('NDCG', config['recall_k'], test_ndcg))
            logging.info('Tst_Loss={:.5f}'.format(test_loss))

            if np.mean(test_recall) >= np.mean(best_recall):
                best_recall = test_recall
                final_test_round = round

            self.save_current_metrics_csv(
                train_losses, test_losses, test_recalls, test_ndcgs, test_hrs,
                generator_losses, config, save_dir, round, final_test_round
            )

        return test_recalls, test_ndcgs, test_hrs, final_test_round

    def save_current_metrics_csv(self, train_losses, test_losses, test_recalls, test_ndcgs, test_hrs,
                                  generator_losses, config, save_dir, current_round, final_test_round):
        import pandas as pd
        
        recall_ks = config['recall_k']
        data = []
        
        for round_idx in range(len(train_losses)):
            row = {
                'Round': round_idx + 1,
                'Train_Loss': train_losses[round_idx],
                'Test_Loss': test_losses[round_idx],
            }
            
            for i, k in enumerate(recall_ks):
                if round_idx < len(test_recalls):
                    row[f'Recall@{k}'] = test_recalls[round_idx][i]
                if round_idx < len(test_ndcgs):
                    row[f'NDCG@{k}'] = test_ndcgs[round_idx][i]
                if round_idx < len(test_hrs):
                    row[f'HR@{k}'] = test_hrs[round_idx][i]
            
            if round_idx < len(generator_losses.get('total', [])):
                row['Generator_Total_Loss'] = generator_losses['total'][round_idx]
                row['Generator_Diversity_Loss'] = generator_losses.get('diversity', [0]*(round_idx+1))[round_idx]
                row['Generator_Contrast_Loss'] = generator_losses.get('contrast', [0]*(round_idx+1))[round_idx]
                row['Generator_Alignment_Loss'] = generator_losses.get('alignment', [0]*(round_idx+1))[round_idx]
            
            data.append(row)
        
        df = pd.DataFrame(data)
        csv_file = f'{save_dir}/training_metrics_round_{current_round}.csv'
        df.to_csv(csv_file, index=False)


def result2str_extended(metric, Ks, results):
    output = []
    for i, k in enumerate(Ks):
        output.append('{}@{} = {:.6f}'.format(metric, k, results[i]))
    return ', '.join(output)
