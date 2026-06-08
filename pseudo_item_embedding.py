import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import logging


class PseudoItemGenerator(nn.Module):

    def __init__(self, embed_size, hidden_size=128, noise_dim=32):
        super(PseudoItemGenerator, self).__init__()
        self.embed_size = embed_size
        self.noise_dim = noise_dim
        self.fc1 = nn.Linear(embed_size + noise_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, embed_size)
        self._init_weights()

    def _init_weights(self):
        for module in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, client_item_embedding, noise=None):
        if client_item_embedding.dim() != 2:
            raise ValueError(f"Expected 2D client item embedding, but got dim {client_item_embedding.dim()}")

        batch_size = client_item_embedding.size(0)

        if noise is None:
            noise = torch.randn(batch_size, self.noise_dim, device=client_item_embedding.device)
        else:
            if noise.dim() == 1:
                noise = noise.unsqueeze(0)
            if noise.size(0) != batch_size:
                raise ValueError(f"Noise batch size mismatch: expected {batch_size}, got {noise.size(0)}")

        x = torch.cat([client_item_embedding, noise], dim=1)
        x = self.relu(self.fc1(x))
        pseudo_item_emb = self.fc2(x)

        return pseudo_item_emb


def diversity_loss(pseudo_items, noises=None):
    P = pseudo_items.size(0)
    
    if P < 2:
        return torch.tensor(0.0, device=pseudo_items.device, requires_grad=True)
    
    pseudo_items_norm = F.normalize(pseudo_items, p=2, dim=1)
    sim_matrix = torch.mm(pseudo_items_norm, pseudo_items_norm.t())
    triu_indices = torch.triu_indices(P, P, offset=1, device=pseudo_items.device)
    pairwise_similarities = sim_matrix[triu_indices[0], triu_indices[1]]
    loss = pairwise_similarities.mean()
    
    return loss


def contrastive_loss(pos_pseudo_items, neg_pseudo_items_dict, client_preference_proxy, 
                     client_idx, tau=0.1):
    device = client_preference_proxy.device
    
    if client_preference_proxy.dim() == 1:
        client_preference_proxy = client_preference_proxy.unsqueeze(0)
    
    if isinstance(pos_pseudo_items, list):
        if len(pos_pseudo_items) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        pos_items = []
        for item in pos_pseudo_items:
            if item.dim() == 1:
                pos_items.append(item.unsqueeze(0))
            else:
                pos_items.append(item)
        pos_items_tensor = torch.cat(pos_items, dim=0)
    else:
        pos_items_tensor = pos_pseudo_items
    
    neg_items_list = []
    for other_idx, other_items in neg_pseudo_items_dict.items():
        if other_idx == client_idx:
            continue
        for item in other_items:
            if isinstance(item, torch.Tensor):
                if item.dim() == 1:
                    neg_items_list.append(item.unsqueeze(0))
                else:
                    neg_items_list.append(item)
    
    if len(neg_items_list) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    neg_items_tensor = torch.cat(neg_items_list, dim=0).to(device)
    
    pos_items_norm = F.normalize(pos_items_tensor, p=2, dim=1)
    neg_items_norm = F.normalize(neg_items_tensor, p=2, dim=1)
    proxy_norm = F.normalize(client_preference_proxy, p=2, dim=1)
    
    pos_similarities = torch.sum(pos_items_norm * proxy_norm, dim=1)
    neg_similarities = torch.sum(neg_items_norm * proxy_norm, dim=1)
    
    pos_exp = torch.exp(pos_similarities / tau)
    numerator = pos_exp.sum()
    
    neg_exp = torch.exp(neg_similarities / tau)
    denominator = numerator + neg_exp.sum()
    
    loss = -torch.log(numerator / (denominator + 1e-8))
    
    return loss


def alignment_loss(generator, client_item_emb, global_params, local_params, noises, verbose=False):
    device = client_item_emb.device
    
    if client_item_emb.dim() == 1:
        client_item_emb = client_item_emb.unsqueeze(0)
    
    if 'network_params_for_alignment' in local_params:
        local_network_params = local_params['network_params_for_alignment']
    else:
        local_network_params = {k: v for k, v in local_params.items()
                                if k.startswith('fc_layers') or k.startswith('affine_output')}
    
    if 'embedding_user.weight' not in local_params:
        return torch.tensor(0.01, device=device, requires_grad=True)
    
    user_emb = local_params['embedding_user.weight'].to(device)
    
    global_network_params = {}
    for key in global_params:
        if key.startswith('fc_layers') or key.startswith('affine_output'):
            global_network_params[key] = global_params[key]
    
    if not global_network_params:
        global_network_params = local_network_params
    
    def forward_pass(item_emb, user_emb, network_params):
        vector = torch.cat([user_emb, item_emb], dim=-1)
        n_layers = sum(1 for k in network_params.keys() if 'fc_layers' in k and 'weight' in k)
        
        for j in range(n_layers):
            weight_key = f'fc_layers.{j}.weight'
            bias_key = f'fc_layers.{j}.bias'
            
            if weight_key in network_params and bias_key in network_params:
                weight = network_params[weight_key].to(device)
                bias = network_params[bias_key].to(device)
                vector = torch.matmul(vector, weight.t()) + bias
                vector = F.leaky_relu(vector)
        
        if 'affine_output.weight' in network_params and 'affine_output.bias' in network_params:
            weight = network_params['affine_output.weight'].to(device)
            bias = network_params['affine_output.bias'].to(device)
            logits = torch.matmul(vector, weight.t()) + bias
            return logits
        
        return vector
    
    def kl_divergence_binary(p_logits, q_logits):
        p = torch.sigmoid(p_logits)
        q = torch.sigmoid(q_logits)
        eps = 1e-8
        p = torch.clamp(p, eps, 1 - eps)
        q = torch.clamp(q, eps, 1 - eps)
        kl = p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
        return kl.mean()
    
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    num_pseudo = 0
    
    try:
        reference_logits = forward_pass(client_item_emb, user_emb, local_network_params)
    except Exception as e:
        return torch.tensor(0.01, device=device, requires_grad=True)
    
    for i, noise in enumerate(noises):
        try:
            noise_input = noise.unsqueeze(0) if noise.dim() == 1 else noise
            pseudo_item = generator(client_item_emb, noise_input)
            
            local_pred_logits = forward_pass(pseudo_item, user_emb, local_network_params)
            global_pred_logits = forward_pass(pseudo_item, client_item_emb, global_network_params)
            
            kl_term1 = kl_divergence_binary(local_pred_logits, global_pred_logits)
            kl_term2 = kl_divergence_binary(local_pred_logits, reference_logits)
            
            pseudo_loss = kl_term1 + kl_term2
            
            if not (torch.isnan(pseudo_loss) or torch.isinf(pseudo_loss)):
                total_loss = total_loss + pseudo_loss
                num_pseudo += 1
                    
        except Exception as e:
            continue
    
    if num_pseudo > 0:
        return total_loss / num_pseudo
    else:
        return torch.tensor(0.01, device=device, requires_grad=True)


def generate_diverse_noises(n_samples, noise_dim, device='cpu', min_distance=0.1, historical_noises=None):
    if historical_noises is None:
        historical_noises = []
    
    noises = []
    max_attempts = 100
    
    for i in range(n_samples):
        attempts = 0
        best_noise = None
        best_min_dist = 0
        
        while attempts < max_attempts:
            attempts += 1
            new_noise = torch.randn(1, noise_dim, device=device)
            
            all_distances = []
            for existing_noise in noises:
                dist = torch.norm(new_noise - existing_noise, p=2).item()
                all_distances.append(dist)
            
            min_dist = min(all_distances) if all_distances else float('inf')
            
            if min_dist >= min_distance:
                noises.append(new_noise)
                break
            
            if best_noise is None or min_dist > best_min_dist:
                best_noise = new_noise.clone()
                best_min_dist = min_dist
        else:
            if best_noise is not None:
                noises.append(best_noise)
            else:
                noises.append(torch.randn(1, noise_dim, device=device))
    
    updated_historical_noises = historical_noises.copy()
    for noise in noises:
        updated_historical_noises.append(noise.cpu())
    
    noises_tensor = torch.cat(noises, dim=0)
    return noises_tensor, updated_historical_noises


def train_pseudo_item_generator(
    generator, 
    optimizer, 
    client_models, 
    server_model_params,
    diverse_noises,
    n_epochs=2, 
    lambda1=0.5,
    lambda2=0.5,
    tau=0.1,
    device='cpu', 
    verbose=False, 
    batch_size=128
):
    generator.train()
    
    all_epoch_losses = {
        'total': [],
        'contrastive': [],
        'alignment': [],
        'diversity': [],
        'fc1_grad_norm': [],
        'fc2_grad_norm': [],
        'fc1_weight_norm': [],
        'fc2_weight_norm': []
    }
    
    collected_pseudo_items = {}
    
    client_indices = list(client_models.keys())
    valid_client_indices = []
    for client_idx in client_indices:
        if client_idx in server_model_params.get('client_item_embeddings', {}):
            valid_client_indices.append(client_idx)
    
    num_clients = len(valid_client_indices)
    
    if num_clients == 0:
        return generator, {k: 0.0 for k in all_epoch_losses.keys()}, {}
    
    for epoch in range(n_epochs):
        shuffled_clients = valid_client_indices.copy()
        random.shuffle(shuffled_clients)
        
        client_batches = []
        for i in range(0, len(shuffled_clients), batch_size):
            client_batches.append(shuffled_clients[i:i + batch_size])
        
        epoch_losses = {
            'total': [],
            'contrastive': [],
            'alignment': [],
            'diversity': []
        }
        
        fc1_grad_norm = 0.0
        fc2_grad_norm = 0.0
        
        for batch_idx, batch_client_ids in enumerate(client_batches):
            optimizer.zero_grad()
            
            batch_pseudo_items = {}
            
            for client_idx in batch_client_ids:
                client_item_emb = server_model_params['client_item_embeddings'][client_idx]['avg_embedding'].to(device)
                if client_item_emb.dim() == 1:
                    client_item_emb = client_item_emb.unsqueeze(0)
                
                client_pseudo_items = []
                for noise_idx in range(diverse_noises.size(0)):
                    noise = diverse_noises[noise_idx:noise_idx + 1]
                    pseudo_item = generator(client_item_emb, noise)
                    client_pseudo_items.append(pseudo_item)
                    
                    if client_idx not in collected_pseudo_items:
                        collected_pseudo_items[client_idx] = []
                    collected_pseudo_items[client_idx].append(pseudo_item.detach().cpu())
                
                batch_pseudo_items[client_idx] = client_pseudo_items
            
            batch_total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            batch_cl_loss = torch.tensor(0.0, device=device)
            batch_align_loss = torch.tensor(0.0, device=device)
            batch_div_loss = torch.tensor(0.0, device=device)
            valid_clients_in_batch = 0
            
            for client_idx in batch_client_ids:
                client_item_emb = server_model_params['client_item_embeddings'][client_idx]['avg_embedding'].to(device)
                if client_item_emb.dim() == 1:
                    client_item_emb = client_item_emb.unsqueeze(0)
                
                client_pseudo_items = batch_pseudo_items[client_idx]
                pseudo_items_tensor = torch.cat(client_pseudo_items, dim=0)
                
                try:
                    div_loss = diversity_loss(pseudo_items_tensor)
                except:
                    div_loss = torch.tensor(0.0, device=device)
                
                try:
                    neg_items_dict = {k: v for k, v in batch_pseudo_items.items() if k != client_idx}
                    cl_loss = contrastive_loss(
                        pos_pseudo_items=client_pseudo_items,
                        neg_pseudo_items_dict=neg_items_dict,
                        client_preference_proxy=client_item_emb,
                        client_idx=client_idx,
                        tau=tau
                    )
                except:
                    cl_loss = torch.tensor(0.0, device=device)
                
                try:
                    align_loss = alignment_loss(
                        generator=generator,
                        client_item_emb=client_item_emb,
                        global_params=server_model_params,
                        local_params=client_models[client_idx],
                        noises=diverse_noises,
                        verbose=False
                    )
                except:
                    align_loss = torch.tensor(0.0, device=device)
                
                client_loss = cl_loss + lambda1 * align_loss + lambda2 * div_loss
                
                if not (torch.isnan(client_loss) or torch.isinf(client_loss)):
                    batch_total_loss = batch_total_loss + client_loss
                    batch_cl_loss = batch_cl_loss + cl_loss.detach()
                    batch_align_loss = batch_align_loss + align_loss.detach()
                    batch_div_loss = batch_div_loss + div_loss.detach()
                    valid_clients_in_batch += 1
            
            if valid_clients_in_batch > 0:
                batch_total_loss = batch_total_loss / valid_clients_in_batch
                batch_total_loss.backward()
                
                fc1_grad_norm = generator.fc1.weight.grad.norm().item() if generator.fc1.weight.grad is not None else 0.0
                fc2_grad_norm = generator.fc2.weight.grad.norm().item() if generator.fc2.weight.grad is not None else 0.0
                
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
                optimizer.step()
                
                epoch_losses['total'].append(batch_total_loss.item())
                epoch_losses['contrastive'].append((batch_cl_loss / valid_clients_in_batch).item())
                epoch_losses['alignment'].append((batch_align_loss / valid_clients_in_batch).item())
                epoch_losses['diversity'].append((batch_div_loss / valid_clients_in_batch).item())
        
        for key in ['total', 'contrastive', 'alignment', 'diversity']:
            if epoch_losses[key]:
                all_epoch_losses[key].append(sum(epoch_losses[key]) / len(epoch_losses[key]))
        
        all_epoch_losses['fc1_weight_norm'].append(generator.fc1.weight.norm().item())
        all_epoch_losses['fc2_weight_norm'].append(generator.fc2.weight.norm().item())
        all_epoch_losses['fc1_grad_norm'].append(fc1_grad_norm)
        all_epoch_losses['fc2_grad_norm'].append(fc2_grad_norm)
    
    avg_losses = {}
    for key in all_epoch_losses:
        if all_epoch_losses[key]:
            avg_losses[key] = sum(all_epoch_losses[key]) / len(all_epoch_losses[key])
        else:
            avg_losses[key] = 0.0
    
    avg_losses['total'] = avg_losses.get('total', 0.0)
    avg_losses['contrast'] = avg_losses.get('contrastive', 0.0)
    avg_losses['alignment'] = avg_losses.get('alignment', 0.0)
    avg_losses['diversity'] = avg_losses.get('diversity', 0.0)
    
    return generator, avg_losses, collected_pseudo_items


def train_pseudo_item_generator_batch_new(
    generator, optimizer, client_models, server_model_params,
    pre_generated_pseudo_items, diverse_noises,
    n_epochs=2, diversity_weight=0.5, contrast_weight=1.0,
    alignment_weight=0.5,
    device='cpu', verbose=False, batch_size=128,
):
    return train_pseudo_item_generator(
        generator=generator,
        optimizer=optimizer,
        client_models=client_models,
        server_model_params=server_model_params,
        diverse_noises=diverse_noises,
        n_epochs=n_epochs,
        lambda1=alignment_weight,
        lambda2=diversity_weight,
        tau=0.1,
        device=device,
        verbose=verbose,
        batch_size=batch_size
    )
