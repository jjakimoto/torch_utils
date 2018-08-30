import numpy as np


def is_update_model(model, old_model):
    """Check if model is updated

    Parameters
    ----------
    model: torch.nn.Module instance
        Model with updated parameters
    old_model: torch.nn.Module instance
        Model with not updated parameters

    Returns
    -------
    bool: If any parameters fail to update, return False
    """
    updated_list = []
    for old_param, param in zip(old_model.parameters(), model.parameters()):
        updated_list.append(
            not np.allclose(old_param.data.numpy(), param.data.numpy()))
    return np.alltrue(updated_list)


def is_same_model(model1, model2):
    """Check if model1 and model 2 are the same

        Parameters
        ----------
        model1: torch.nn.Module instance
        model2: torch.nn.Module instance

        Returns
        -------
        bool: If any pairs parameters are different, return False
    """
    same_list = []
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        same_list.append(np.allclose(param1.data.numpy(), param2.data.numpy()))
    return np.alltrue(same_list)