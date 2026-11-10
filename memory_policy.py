"""Optional residual GRU; the original CNN and its checkpoint remain unchanged."""
import torch
from torch import nn
from policy import ActorCritic


class MemoryActorCritic(ActorCritic):
    is_recurrent = True

    def __init__(self, n_actions=18, vec_dim=3, stack=7, lags=None, memory_size=128):
        super().__init__(n_actions, vec_dim, stack, lags)
        self.memory_size = int(memory_size)
        self.memory = nn.GRUCell(256, self.memory_size)
        self.memory_projection = nn.Linear(self.memory_size, 256)
        # Exact CNN behaviour at warm start, with a trainable memory residual.
        nn.init.zeros_(self.memory_projection.weight)
        nn.init.zeros_(self.memory_projection.bias)

    def initial_state(self, batch, device=None):
        return self.pi.weight.new_zeros(batch, self.memory_size, device=device)

    def forward(self, obs):
        raise RuntimeError("Recurrent policy requires forward_step or forward_sequence; do not silently discard memory")

    def forward_step(self, obs, state, episode_start=None):
        if state is None:
            state = self.initial_state(obs['vec'].shape[0])
        if episode_start is not None:
            state = state * (~episode_start.bool()).to(state.dtype)[:, None]
        features = self.hidden(obs)
        state = self.memory(features, state)
        features = features + self.memory_projection(state)
        return self.pi(features), self.v(features).squeeze(-1), state

    def forward_sequence(self, obs, state, episode_start):
        """Time-major [T,B,...], preserving ordering and resetting each episode."""
        length, batch = obs['vec'].shape[:2]
        encoded = self.hidden({k: v.reshape(length * batch, *v.shape[2:]) for k,v in obs.items()})
        encoded = encoded.reshape(length, batch, -1)
        if state is None:
            state = self.initial_state(batch)
        memories = []
        for t in range(length):
            state = state * (~episode_start[t].bool()).to(state.dtype)[:, None]
            state = self.memory(encoded[t], state)
            memories.append(state)
        features = encoded + self.memory_projection(torch.stack(memories))
        return self.pi(features), self.v(features).squeeze(-1), state


def build_policy(checkpoint, device='cuda'):
    """One explicit factory, including old CNN checkpoints without an arch tag."""
    arch = checkpoint.get('arch', 'cnn')
    if arch == 'cnn_gru':
        net = MemoryActorCritic(checkpoint.get('n_actions',18), checkpoint.get('vec_dim',3),
                               checkpoint.get('stack',4), checkpoint.get('lags'),
                               checkpoint.get('memory_size',128))
    elif arch == 'cnn':
        net = ActorCritic(checkpoint.get('n_actions',18), checkpoint.get('vec_dim',3),
                          checkpoint.get('stack',4), checkpoint.get('lags'))
    else:
        raise ValueError(f'Unsupported policy architecture: {arch}')
    net = net.to(device).to(memory_format=torch.channels_last)
    net.load_state_dict(checkpoint['net'])
    return net


def step_policy(net, obs, state=None, episode_start=None):
    if getattr(net,'is_recurrent',False):
        return net.forward_step(obs,state,episode_start)
    logits, value = net(obs)
    return logits,value,None


def sequence_policy(net, obs, state, episode_start):
    if getattr(net,'is_recurrent',False):
        return net.forward_sequence(obs,state,episode_start)
    length,batch=obs['vec'].shape[:2]
    logits,value=net({k:v.reshape(length*batch,*v.shape[2:]) for k,v in obs.items()})
    return logits.reshape(length,batch,-1),value.reshape(length,batch),None
