import torch


def train_step(model, train_loader, optimizer,
               loss_func, score_func=None,
               epoch=None, device='cpu', log_interval=100,
               silent=False, **kwargs):
    '''Train step for one epoch

    Parameters
    ----------
    model: nn.Module subclass
    train_loader: Defined by torch.utils.data.DataLoader
    optimizer: optimizer of torch.optim
    epoch: int, optional
        Used for display
    loss_func: function of torch.nn.functional
        loss function to optimize
    score_func: score function, optional
        e.g., sklearn.metrics.accuracy_score
    device: str, (default 'cpu')
    log_interval: int, (default 100)
        How frequent to display
    silent: bool, (default False)
        If True, not display the result
    kwargs: optional
        parameters used for loss function
    '''
    model.train()
    loss_total = 0.
    score_total = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target, **kwargs)
        loss.backward()
        optimizer.step()
        if not silent:
            loss = loss_func(output, target, size_average=False)
            loss_total += loss.item() * len(data)
            if score_func is not None:
                pred = model.predict(data)
                score = score_func(target, pred)
                score_total += score * len(data)
            if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
                print(f'Train Epoch: {epoch} '
                      f'[{epoch, batch_idx * len(data)}/{len(train_loader.dataset):.4g} '
                      f'({100. * batch_idx / len(train_loader):.4g}%)]\tLoss: {loss.item():.4g}')
                if score_func is not None:
                    print(f'Score: {score:.4g}\n')
                else:
                    print()
    # Display the result of entire epoch
    if not silent:
        mean_loss = loss_total / len(train_loader.dataset)
        print(f'Average Loss: {mean_loss:.4g}')
        if score_func is not None:
            mean_score = score_total / len(train_loader.dataset)
            print(f'Average Score: {mean_score:.4g}\n')
        else:
            print()


def test_step(model, test_loader, loss_func,
              score_func=None, device='cpu', **kwargs):
    '''Test step for one epoch

    Parameters
    ----------
    model: nn.Module subclass
    test_loader: Defined by torch.utils.data.DataLoader
    loss_func: function of torch.nn.functional
        loss function to optimize
    score_func: score function, optional
        e.g., sklearn.metrics.accuracy_score
    device: str, (default 'cpu')
    kwargs: optional
        parameters used for loss function
    '''
    model.eval()
    test_loss = 0
    score = 0.
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target,
                                   size_average=False,
                                   **kwargs).item()
            if score_func is not None:
                pred = model.predict(data)
                score += score_func(target, pred) * len(data)

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set')
    print(f'Average loss: {test_loss:.4g}')
    if score_func is not None:
        mean_score = score / len(test_loader.dataset)
        print(f'Average Score: {mean_score:.4g}\n')
    else:
        print()
