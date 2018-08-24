# PyTorch Utils
Utils functions for PyTorch

# Installation
In this directory, execute

`python setup.py install`

After installation, you can use the library by
`import torch_utils`

`
# Examples
```python
import torch.optim as optim
import torch.utils.data as tdata

from torchvision import datasets
import torchvision.transforms as transforms

from torch_utils.training import train_step, test_step


batch_size = 32

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

train_loader = tdata.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transform),
        batch_size=batch_size, shuffle=True)
    
test_loader = tdata.DataLoader(
        datasets.MNIST('../data', train=False, transform=transform),
        batch_size=batch_size, shuffle=True)

# PyTorch model inherited from torch.nn.Module
model = Net()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# Need 
for i in range(100):
    train_step(model, train_loader, optimizer,
               epoch=i, log_interval=1000)
    test_step(model, test_loader) 
```