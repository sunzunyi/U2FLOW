import torch

def build_network(cfg):
    name = cfg.network
    if name == 'raft_2f_u':
        from .raft_2f_u import RAFT as network
    else:
        raise ValueError(f"Network = {name} is not a valid optimizer!")

    return network(cfg[name])
