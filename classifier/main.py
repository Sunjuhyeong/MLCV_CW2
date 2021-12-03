import torch
import torch.nn as nn
import torchvision.datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from classifier.model import CNNClassifier

from tqdm import trange



if __name__ == '__main__':
    transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])

    mnist_train = torchvision.datasets.MNIST(root='data/',
                                             train=True,
                                             transform=transforms,
                                             download=True)

    mnist_test = torchvision.datasets.MNIST(root='data/',
                                            train=False,
                                            transform=transforms,
                                            download=True)

    train_loader = DataLoader(dataset=mnist_train,
                              batch_size=8,
                              shuffle=True,
                              drop_last=True)

    test_loader = DataLoader(dataset=mnist_test,
                             batch_size=8,
                             shuffle=False,
                             drop_last=False)

    model = CNNClassifier()
    model = model.cuda(0)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    best_accuracy = 0.
    for epoch in trange(200):

        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for step, (x, y) in pbar:
            x, y = x.cuda(0), y.cuda(0)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        correct = 0
        model.eval()
        for (x, y) in test_loader:
            x, y = x.cuda(0), y.cuda(0)

            with torch.no_grad():
                logits = model(x)
                correct += (torch.argmax(logits, 1) == y).sum()

        accuracy = correct / len(mnist_test)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f'checkpoints_test/best.pt')
            print(f'[Epoch : {epoch}/200] Best Accuracy : {accuracy:.6f}%')
