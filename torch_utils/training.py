import torch
import torch.nn.functional as F


def train_step(model, train_loader, optimizer, epoch=None,
               loss_func=F.nll_loss, device='cpu', log_interval=100,
               **kwargs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target, **kwargs)
        loss.backward()
        optimizer.step()
        if not log_interval and batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} \
                   [{epoch, batch_idx * len(data)}/{len(train_loader.dataset)} \
                   ({100. * batch_idx / len(train_loader)}%)]\tLoss: {loss.item()}')


def test_step(model, test_loader, loss_func=F.nll_loss,
              device='cpu', **kwargs):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target,
                                   size_average=False,
                                   **kwargs).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss}, \
          Accuracy: {correct}/{len(test_loader.dataset)} \
          ({100. * correct / len(test_loader.dataset)}%)\n')
