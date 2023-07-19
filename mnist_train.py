import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import model.FFN as FFN

# hyperparameter
BATCH_SIZE = 512
EPOCHS = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 3e-4

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(data.size()[0], -1)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _ = model(data, True)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if(batch_idx+1)%30 == 0: 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size()[0], -1)
            data, target = data.to(device), target.to(device)
            output, _ = model(data, True)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True, 
                       transform=transforms.Compose( [transforms.ToTensor(),
#                                                      transforms.Normalize((0.1307,), (0.3081,))
                                                      ] 
                                                    )
                       ),
        batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=False, 
                       transform=transforms.Compose( [transforms.ToTensor(),
#                                                      transforms.Normalize((0.1307,), (0.3081,))
                                                      ] 
                                                    )
                       ),
        batch_size=BATCH_SIZE, shuffle=True)
    
    input_features = 28 * 28
    #hidden_layers = [256, 128, 10]
    hidden_layers = [1024, 256, 64, 10]
    model = FFN.FFN(input_features, list(hidden_layers)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(1, EPOCHS + 1):
        train(model, DEVICE, train_loader, optimizer, epoch)
        test(model, DEVICE, test_loader)
    
    torch.save(model, "./data/mnist/ffn_4layer_not_norm.pth")
    
if __name__ == "__main__":
    main()
