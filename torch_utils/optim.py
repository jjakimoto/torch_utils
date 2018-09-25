from copy import deepcopy
import torch.optim as optim


def get_optimizer(parameters, lr_spec, scheduler_spec=None):
    lr_spec = deepcopy(lr_spec)
    name = lr_spec['name']
    del lr_spec['name']
    if name == 'adam':
        optimizer = optim.Adam(parameters, **lr_spec)
    elif name == 'rmsp':
        optimizer = optim.RMSprop(parameters, **lr_spec)
    else:
        raise NotImplementedError(f'No Implementation for name={name}')
    if scheduler_spec is None:
        return optimizer
    else:
        name = scheduler_spec['name']
        del scheduler_spec['name']
        if name == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_spec)
        elif name == 'exponential':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_spec)
        elif name == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_spec)
        else:
            raise NotImplementedError(f'No Implementation for name={name}')
        return optimizer, scheduler