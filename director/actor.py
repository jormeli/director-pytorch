import torch 
import torch.nn as nn
import numpy as np

class DiscreteActionModel(nn.Module):
    def __init__(
        self,
        action_size,
        deter_size,
        stoch_size,
        embedding_size,
        actor_info,
        expl_info,
        goal_vae_fn,
        goal_vae_latents,
        goal_vae_classes,
    ):
        super().__init__()
        self.goal_vae_fn = goal_vae_fn
        self.goal_vae_size = goal_vae_latents * goal_vae_classes
        self.action_size = action_size
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.embedding_size = embedding_size
        self.layers = actor_info['layers']
        self.node_size = actor_info['node_size']
        self.act_fn = actor_info['activation']
        self.dist = actor_info['dist']
        self.act_fn = actor_info['activation']
        self.train_noise = expl_info['train_noise']
        self.eval_noise = expl_info['eval_noise']
        self.expl_min = expl_info['expl_min']
        self.expl_decay = expl_info['expl_decay']
        self.expl_type = expl_info['expl_type']
        self.model = self._build_model()

    def _build_model(self):
        model = [nn.Linear(self.goal_vae_size, self.node_size)]
        model += [self.act_fn()]
        for i in range(1, self.layers):
            model += [nn.Linear(self.node_size, self.node_size)]
            model += [self.act_fn()]

        if self.dist == 'one_hot':
            model += [nn.Linear(self.node_size, self.action_size)]
        else:
            raise NotImplementedError
        return nn.Sequential(*model) 

    def forward(self, model_state):
        with torch.no_grad():
            discrete_state, _ = self.goal_vae_fn(model_state)
        action_dist = self.get_action_dist(discrete_state)
        action = action_dist.sample()
        action = action + action_dist.probs - action_dist.probs.detach()
        return action, action_dist

    def get_action_dist(self, modelstate):
        logits = self.model(modelstate)
        if self.dist == 'one_hot':
            return torch.distributions.OneHotCategorical(logits=logits)         
        else:
            raise NotImplementedError
            
    def add_exploration(self, action: torch.Tensor, itr: int, mode='train'):
        if mode == 'train':
            expl_amount = self.train_noise
            expl_amount = expl_amount - itr/self.expl_decay
            expl_amount = max(self.expl_min, expl_amount)
        elif mode == 'eval':
            expl_amount = self.eval_noise
        else:
            raise NotImplementedError
            
        if self.expl_type == 'epsilon_greedy':
            if np.random.uniform(0, 1) < expl_amount:
                index = torch.randint(0, self.action_size, action.shape[:-1], device=action.device)
                action = torch.zeros_like(action)
                action[:, index] = 1
            return action

        raise NotImplementedError
