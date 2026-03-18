
def get_loss(cfg):
    if cfg.type == 'unFlowLoss_Raft':
        from.flow_loss import unFlowLoss_Raft
        loss = unFlowLoss_Raft(cfg)
    else:
        raise NotImplementedError(cfg.type)
    return loss
