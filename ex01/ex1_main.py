import argparse
import os
import time

import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import optim

from my_models_skeleton import ViT, CrossViT   # rename the skeleton file for your implementation / comment before testing for ResNet

# from test import ViT, CrossViT

def device_choose():
    if torch.cuda.is_available():
        device_name = 'cuda'
    elif torch.backends.mps.is_available():
        device_name = 'mps'
    else:
        device_name = 'cpu'
    print(f'Using {device_name}')
    return device_name
#
def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network to classify CIFAR10')
    # parser.add_argument('--model', type=str, default='r18', help='model to train (default: r18)')
    parser.add_argument('--model', type=str, default='cvit', help='model to train (default: r18)')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    # Specifies the momentum factor for the SGD (Stochastic Gradient Descent) optimizer.
    # Momentum helps accelerate training by moving past small gradients.
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    # Sets the random seed for reproducibility, ensuring that results are consistent across different runs.
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    # Specifies how many batches to wait before printing the training status (i.e., loss). Helps in monitoring training progress.
    parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    # If specified, performs a quick pass through the data to ensure everything works correctly, without actually training the model.
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    return parser.parse_args()

def train(model, trainloader, optimizer, criterion, device, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        # load data and target to device
        data, target = data.to(device), target.to(device)
        # Clear the gradients of all optimized tensors before each training step.
        # By default, gradients accumulate in Pytorch
        optimizer.zero_grad()
        # do a forward pass through the model with current batch of data
        output = model(data)
        # calculate loss and normalize
        loss = criterion(output, target)/len(output)
        # backward propagation and compute the gradients of the loss respect to the model parameters
        # and save to parameter.grad
        loss.backward()
        # update the parameter with parameter.grad
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                       100. * batch_idx / len(trainloader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader, criterion, set="Test"):
    model.eval()
    test_loss = 0
    correct = 0
    # disable gradient calculation, this reduces memory usage and speeds up(gradients are not needed here)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss, item() get scalar value from pytorch tensor
            test_loss += criterion(output, target).item()
            # dim = 1, get the class index with the highest probability
            # with keepdim, the shape remains [batch_size, 1], otherwise it will be [batch_size]
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # .view_as() reshape the target to the pre.shape
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    # calculate accuracy of current model
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        set, test_loss, correct, len(test_loader.dataset),
        accuracy))
    return accuracy

def run(args):
    # Download and load the training data
    # Task 1.1 add randomcrop, randomhorizontalflip
    transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    # randomly crop to size 32*32 with 4 piex padding
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    # ImageNet mean/std values should also fit okayish for CIFAR
									transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) ) 
                                    ])

    # TODO: adjust folder
    dataset = datasets.CIFAR10('./cifar10/folder', download=True, train=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(dataset,
                                                     [int(len(dataset)*0.9), len(dataset)-int(len(dataset)*0.9)])
    # trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    # valloader = DataLoader(valset, batch_size=64, shuffle=False)
    trainloader = DataLoader(trainset, batch_size= args.batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size= args.batch_size, shuffle=False)

    # Download and load the test data
    # TODO: adjust folder
    testset = datasets.CIFAR10('./cifar10/folder/', download=True, train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True)

    # Build a feed-forward network
    print(f"Using {args.model}")
    if args.model == "r18":
        model = models.resnet18(pretrained=False)
    elif args.model == "vit":
        model = ViT(image_size = 32, patch_size = 8, num_classes = 10, dim = 64,
                    depth = 2, heads = 8, mlp_dim = 128, dropout = 0.1,
                    emb_dropout = 0.1) 
    elif args.model == "cvit":
        model = CrossViT(image_size = 32, num_classes = 10, sm_dim = 64, 
                         lg_dim = 128, sm_patch_size = 8, sm_enc_depth = 2,
                         sm_enc_heads = 8, sm_enc_mlp_dim = 128, 
                         sm_enc_dim_head = 64, lg_patch_size = 16, 
                         lg_enc_depth = 2, lg_enc_heads = 8, 
                         lg_enc_mlp_dim = 128, lg_enc_dim_head = 64,
                         cross_attn_depth = 2, cross_attn_heads = 8,
                         cross_attn_dim_head = 64, depth = 3, dropout = 0.1,
                         emb_dropout = 0.1)

    # Define the loss, using cross entropy loss with reduction set to "sum"
    criterion = nn.CrossEntropyLoss(reduction="sum")
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")
    device = torch.device(device_choose())
    model.to(device)
    # using stochastic gradient descent with specified learning rate and momentum
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Task 1.1 save the model with the highest validation accuracy
    hightest_accuracy = 0.0
    best_epoch = -1
    os.makedirs('./models', exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train(model, trainloader, optimizer, criterion, device, epoch)
        _accuracy = test(model, device, valloader, criterion, set="Validation")
        if _accuracy > hightest_accuracy:
            hightest_accuracy = _accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), f"./models/{args.model}.pth")
    print(f"current best accuracy is {hightest_accuracy}% from epoch {best_epoch}")
    test(model, device, testloader, criterion)

if __name__ == '__main__':
    # choose
    # device = torch.device(device_choose())
    args = parse_args()
    s = time.time()
    run(args)
    e = time.time()
    print(f"{e - s}s used")
