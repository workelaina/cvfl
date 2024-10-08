"""
Train VFL on ModelNet-10 dataset
"""

import argparse
import numpy as np
import time
import os
import random
import pickle
import math
import sys
# import latbin

# import pandas as pd
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

from network import Server_L2, myResNet

pgdcfg_eps = 0.3
pgdcfg_sigma = 0.15

MVCNN = 'mvcnn'
RESNET = 'resnet'
MODELS = [RESNET,MVCNN]

def topk(tensor, compress_ratio):
    """
    Get topk elements in tensor
    """
    shape = tensor.shape
    tensor = tensor.flatten()
    k = max(1, int(tensor.numel() * compress_ratio))
    _, indices = torch.topk(tensor.abs(), k, sorted=False,)
    values = torch.gather(tensor, 0, indices)
    numel = tensor.numel()
    tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
    tensor_decompressed.scatter_(0, indices, values)
    return tensor_decompressed.view(shape)

# Set up input arguments
num_clients = int(sys.argv[3])

parser = argparse.ArgumentParser(description='MVCNN-PyTorch')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--num_clients', type=int, help='Number of clients to split data between vertically',
                        default=2)
parser.add_argument('--depth', choices=[18, 34, 50, 101, 152], type=int, metavar='N', default=18, help='resnet depth (default: resnet18)')
parser.add_argument('--model', '-m', metavar='MODEL', default=RESNET, choices=MODELS,
                    help='pretrained model: ' + ' | '.join(MODELS) + ' (default: {})'.format(RESNET))
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run (default: 100)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate (default: 0.0001)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--lr-decay-freq', default=30, type=float,
                    metavar='W', help='learning rate decay (default: 30)')
parser.add_argument('--lr-decay', default=0.1, type=float,
                    metavar='W', help='learning rate decay (default: 0.1)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--local_epochs', type=int, help='Number of local epochs to run at each client before synchronizing',
                    default=1)
parser.add_argument('--quant_level', type=int, help='Number of quantization buckets',
                    default=0)
parser.add_argument('--vecdim', type=int, help='Vector quantization dimension',
                    default=1)
parser.add_argument('--comp', type=str, help='Which compressor', default="")
parser.add_argument('--seed', type=int, help='Random seed to use', default=42)

# Parse input arguments
args = parser.parse_args()

suffix = f"_cifar_NC{args.num_clients}_LE{args.local_epochs}_quant{args.quant_level}_dim{args.vecdim}_comp{args.comp}_seed{args.seed}"

np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

assert torch.cuda.is_available()
device = torch.device("cuda:0")

def quantize_vector(x, quant_min=0, quant_max=1, quant_level=5, dim=2):
    raise ValueError('latbin')
    return None
#     """Uniform vector quantization approach

#     Notebook: C2S2_DigitalSignalQuantization.ipynb

#     Args:
#         x: Original signal
#         quant_min: Minimum quantization level
#         quant_max: Maximum quantization level
#         quant_level: Number of quantization levels
#         dim: dimension of vectors to quantize

#     Returns:
#         x_quant: Quantized signal

#         Currently only works for 2 dimensions and 
#         quant_levels of 4, 8, and 16.
#     """

#     dither = np.random.uniform(-(quant_max-quant_min)/(2*(quant_level-1)), 
#                                 (quant_max-quant_min)/(2*(quant_level-1)),
#                                 size=np.array(x).shape) 
#     # Move into 0,1 range:
#     x_normalize = x/np.max(x)
#     x_normalize = x_normalize + dither

#     A2 = latbin.lattice.ALattice(dim,scale=1/(2*math.log(quant_level,2)))
#     # What?
#     # if quant_level == 4:
#     #     A2 = latbin.lattice.ALattice(dim,scale=1/4)
#     # elif quant_level == 8:
#     #     A2 = latbin.lattice.ALattice(dim,scale=1/8.5)
#     # elif quant_level == 16:
#     #     A2 = latbin.lattice.ALattice(dim,scale=1/19)
    
#     for i in range(0, x_normalize.shape[1], dim):
#         x_normalize[:,i:(i+dim)] = A2.lattice_to_data_space(
#                                         A2.quantize(x_normalize[:,i:(i+dim)]))

#     # Move out of 0,1 range:
#     x_normalize = np.max(x)*(x_normalize - dither)
#     return torch.from_numpy(x_normalize).float().cuda(device)


def quantize_scalar(x, quant_min=0, quant_max=1, quant_level=5):
    """Uniform quantization approach

    Notebook: C2S2_DigitalSignalQuantization.ipynb

    Args:
        x: Original signal
        quant_min: Minimum quantization level
        quant_max: Maximum quantization level
        quant_level: Number of quantization levels

    Returns:
        x_quant: Quantized signal
    """
    x_normalize = np.array(x)

    # Move into 0,1 range:
    x_normalize = x_normalize/np.max(x)
    x_normalize = np.nan_to_num(x_normalize)

    dither = np.random.uniform(-(quant_max-quant_min)/(2*(quant_level-1)),
				(quant_max-quant_min)/(2*(quant_level-1)),
				size=x_normalize.shape)
    x_normalize = x_normalize + dither

    x_normalize = (x_normalize-quant_min) * (quant_level-1) / (quant_max-quant_min)
    x_normalize[x_normalize > quant_level - 1] = quant_level - 1
    x_normalize[x_normalize < 0] = 0
    x_normalize_quant = np.around(x_normalize)
    x_quant = (x_normalize_quant) * (quant_max-quant_min) / (quant_level-1) + quant_min

    # Move out of 0,1 range:
    x_quant = np.max(x)*(x_quant - dither)
    return torch.from_numpy(x_quant).float().cuda(device)

print('Loading data')

transform = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dset_train = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
dset_val = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# class CriteoDataset(Dataset):
#     def __init__(self, features, labels):
#         self.features = features
#         self.labels = labels

#     def __len__(self):
#         return len(self.features)

#     def __getitem__(self, idx):
#         x = self.features[idx]
#         y = self.labels[idx]
#         return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# def load_criteo_data(file_path):
#     data = pd.read_csv(file_path)
#     labels = data.iloc[:, -1].values
#     features = data.iloc[:, :-1]

#     numeric_features = features.select_dtypes(include=['int64', 'float64']).columns
#     categorical_features = features.select_dtypes(include=['object']).columns

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', StandardScaler(), numeric_features),
#             ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
#         ])

#     features = preprocessor.fit_transform(features)

#     return features, labels  


# file_path = os.path.join('data/criteo.csv')
# features, labels = load_criteo_data(file_path)

# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, stratify=labels)

# dset_train = CriteoDataset(X_train, y_train)
# dset_val = CriteoDataset(X_test, y_test)


train_loader = DataLoader(dset_train, batch_size=args.batch_size, shuffle=False, num_workers=1)
test_loader = DataLoader(dset_val, batch_size=args.batch_size, shuffle=False, num_workers=1)

print(dset_train, dset_val)
print(train_loader, test_loader)

losses = []
accs_train = []
accs_test = []

best_acc = 0.0
best_loss = 0.0
start_epoch = 0


print('BEGIN [init model & optim]')
models = []
optimizers = []
# Make models for each client
for i in range(num_clients+1):
    if i == num_clients:
        model = Server_L2(256, 10)
    else:
        model = myResNet(18, 1, 128)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    model.to(device)
    cudnn.benchmark = True

    models.append(model)
    optimizers.append(optimizer)

server_model_comp = Server_L2(256, 10)

server_model_comp.to(device)
server_optimizer_comp = torch.optim.SGD(server_model_comp.parameters(), lr=args.lr)
print('END [init model & optim]')

# Loss and Optimizer
n_epochs = args.epochs
criterion = nn.CrossEntropyLoss()
coords_per = args.batch_size

def save_eval(models, train_loader, test_loader, losses, accs_train, accs_test, step, train_size):
    """
    Evaluate and save current loss and accuracy
    """
    avg_train_acc, avg_loss = eval(models, train_loader)
    avg_test_acc, _ = eval(models, test_loader)

    losses.append(avg_loss)
    accs_train.append(avg_train_acc)
    accs_test.append(avg_test_acc)

    pickle.dump(losses, open(f'./loss{suffix}.pkl','wb'))
    pickle.dump(accs_train, open(f'./accs_train{suffix}.pkl','wb'))
    pickle.dump(accs_test, open(f'./accs_test{suffix}.pkl','wb'))

    print('Iter [%d/%d]: Test Acc: %.2f - Train Acc: %.2f - Loss: %.4f' 
            % (step + 1, train_size, avg_test_acc.item(), avg_train_acc.item(), avg_loss.item()))

def train(models, optimizers, epoch): #, centers):
    """
    Train all clients on all batches 
    """
    global server_model_comp, server_optimizer_comp

    train_size = len(train_loader)
    server_model = models[-1]
    server_optimizer = optimizers[-1]

    Hs = np.empty((len(train_loader), num_clients), dtype=object)
    Hs.fill([])
    grads_Hs = np.empty((num_clients), dtype=object)
    grads_Hs.fill([])

    ratio = 0
    comp = args.comp
    if args.quant_level > 0:
        ratio = math.log(args.quant_level,2)/32

    for step, (inputs, targets) in enumerate(train_loader):
        print(step)
        # Convert from list of 3D to 4D
        inputs = np.stack(inputs, axis=1)

        inputs = torch.from_numpy(inputs)

        inputs, targets = inputs.to(device), targets.to(device)
        # inputs, targets = Variable(inputs), Variable(targets)

        eta = torch.FloatTensor(*inputs.shape).uniform_(-pgdcfg_eps, pgdcfg_eps).to(device)
        for _i in range(40):
            eta.requires_grad_()
            adv_inp = inputs + eta
            grads = list()

            H_orig = [None] * num_clients
            for i in range(num_clients):
                r = math.floor(i/2)
                c = i % 2
                section = coords_per
                x_local = adv_inp[:, section*r:section*(r+1)]
                x_local = torch.transpose(x_local,0,1)
                with torch.no_grad():
                    H_orig[i] = models[i](x_local)

                # Compress embedding
                if comp != "":
                    if comp == "topk" and not (epoch == 0 and step == 0):
                        # Choose top k elements based on grads_Hs[i]
                        H_tmp = H_orig[i].cpu().detach().numpy()
                        num = math.ceil(H_tmp.shape[1]*(1-ratio))
                        grads = np.abs(grads_Hs[i])
                        idx = np.argpartition(grads, num)[:num]
                        indices = idx[np.argsort((grads)[idx])]
                        H_tmp[:,indices[:num]] = 0
                        H_orig[i] = torch.from_numpy(H_tmp).float().cuda(device)
                    elif comp == "topk": 
                        # If first iteration, do nothing
                        pass
                    elif args.vecdim == 1:
                        # Scalar quantization
                        H_orig[i] = quantize_scalar(H_orig[i].cpu().detach().numpy(), 
                            quant_level=args.quant_level)
                    else:
                        # Vector quantization
                        H_orig[i] = quantize_vector(H_orig[i].cpu().detach().numpy(),                        
                                quant_level=args.quant_level, dim=args.vecdim)

            # Compress server model
            if comp != "":
                tmp_dict = server_model.state_dict()
                for key,value in tmp_dict.items():
                    vdim = value.dim() 
                    shape = value.shape
                    if comp == "topk":
                        tmp_dict[key] = topk(value, ratio) 
                    elif args.vecdim == 1:
                        if vdim == 1:
                            value = value.reshape(1,-1)
                        tmp_dict[key] = quantize_scalar(value.cpu().detach().numpy(), 
                            quant_level=args.quant_level).reshape(shape)
                    else:
                        if vdim == 1:
                            value = value.reshape(1,-1)
                        tmp_dict[key] = quantize_vector(value.cpu().detach().numpy(), 
                            quant_level=args.quant_level, dim=args.vecdim).reshape(shape)
                server_model_comp.load_state_dict(tmp_dict)
            else:
                server_model_comp = server_model

            # Train clients
            for i in range(num_clients):
                r = math.floor(i/2)
                c = i % 2
                section = coords_per
                x_local = adv_inp[:, section*r:section*(r+1)]
                x_local = torch.transpose(x_local,0,1)
                H = H_orig.copy()
                model = models[i]
                optimizer = optimizers[i]

                # Calculate number of local iterations
                client_epochs = args.local_epochs
                # Train
                for le in range(client_epochs):
                    # compute output
                    outputs = model(x_local)
                    H[i] = outputs
                    outputs = server_model_comp(torch.cat(H,axis=1))
                    loss = criterion(outputs, targets.to(torch.int64))

                    if le == 0:
                        x_grad = torch.autograd.grad(
                            loss,
                            eta,
                            only_inputs=True,
                            retain_graph=True
                        )[0].detach()
                        grads.append(x_grad[:, section*r:section*(r+1)])

                    # compute gradient and do gradient step
                    optimizer.zero_grad()
                    server_optimizer_comp.zero_grad()
                    loss.backward(retain_graph=True)
                    params = []
                    for param in model.parameters():
                        params.append(param.grad)
                    params[-1] = params[-1].detach().cpu().numpy()
                    grads_Hs[i] = np.array(params[-1])
                    optimizer.step()

            # x_grad = torch.cat(grads)
            x_grad = grads[0] + grads[1]
            adv_inp += x_grad.sign() * pgdcfg_sigma
            adv_inp.clamp_(0., 1.)
            eta = adv_inp - inputs
            eta.clamp_(-pgdcfg_eps, pgdcfg_eps)

            # Train server
            for le in range(args.local_epochs):
                H = H_orig.copy()
                # compute output
                outputs = server_model(torch.cat(H,axis=1))
                loss = criterion(outputs, targets.to(torch.int64))

                # compute gradient and do SGD step
                server_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                server_optimizer.step()

        # Exchange embeddings
        H_orig = [None] * num_clients
        for i in range(num_clients):
            r = math.floor(i/2)
            c = i % 2
            section = coords_per
            x_local = inputs[:, section*r:section*(r+1)]
            x_local = torch.transpose(x_local,0,1)
            with torch.no_grad():
                H_orig[i] = models[i](x_local)

            # Compress embedding
            if comp != "":
                if comp == "topk" and not (epoch == 0 and step == 0):
                    # Choose top k elements based on grads_Hs[i]
                    H_tmp = H_orig[i].cpu().detach().numpy()
                    num = math.ceil(H_tmp.shape[1]*(1-ratio))
                    grads = np.abs(grads_Hs[i])
                    idx = np.argpartition(grads, num)[:num]
                    indices = idx[np.argsort((grads)[idx])]
                    H_tmp[:,indices[:num]] = 0
                    H_orig[i] = torch.from_numpy(H_tmp).float().cuda(device)
                elif comp == "topk": 
                    # If first iteration, do nothing
                    pass
                elif args.vecdim == 1:
                    # Scalar quantization
                    H_orig[i] = quantize_scalar(H_orig[i].cpu().detach().numpy(), 
                        quant_level=args.quant_level)
                else:
                    # Vector quantization
                    H_orig[i] = quantize_vector(H_orig[i].cpu().detach().numpy(),                        
                            quant_level=args.quant_level, dim=args.vecdim)

        # Compress server model
        if comp != "":
            tmp_dict = server_model.state_dict()
            for key,value in tmp_dict.items():
                vdim = value.dim() 
                shape = value.shape
                if comp == "topk":
                    tmp_dict[key] = topk(value, ratio) 
                elif args.vecdim == 1:
                    if vdim == 1:
                        value = value.reshape(1,-1)
                    tmp_dict[key] = quantize_scalar(value.cpu().detach().numpy(), 
                        quant_level=args.quant_level).reshape(shape)
                else:
                    if vdim == 1:
                        value = value.reshape(1,-1)
                    tmp_dict[key] = quantize_vector(value.cpu().detach().numpy(), 
                        quant_level=args.quant_level, dim=args.vecdim).reshape(shape)
            server_model_comp.load_state_dict(tmp_dict)
        else:
            server_model_comp = server_model

        # Train clients
        for i in range(num_clients):
            r = math.floor(i/2)
            c = i % 2
            section = coords_per
            x_local = inputs[:, section*r:section*(r+1)]
            x_local = torch.transpose(x_local,0,1)
            H = H_orig.copy()
            model = models[i]
            optimizer = optimizers[i]

            # Calculate number of local iterations
            client_epochs = args.local_epochs
            # Train
            for le in range(client_epochs):
                # compute output
                outputs = model(x_local)
                H[i] = outputs
                outputs = server_model_comp(torch.cat(H,axis=1))
                loss = criterion(outputs, targets.to(torch.int64))

                # compute gradient and do gradient step
                optimizer.zero_grad()
                server_optimizer_comp.zero_grad()
                loss.backward(retain_graph=True)
                params = []
                for param in model.parameters():
                    params.append(param.grad)
                params[-1] = params[-1].detach().cpu().numpy()
                grads_Hs[i] = np.array(params[-1])
                optimizer.step()

        # Train server
        for le in range(args.local_epochs):
            H = H_orig.copy()
            # compute output
            outputs = server_model(torch.cat(H,axis=1))
            loss = criterion(outputs, targets.to(torch.int64))

            # compute gradient and do SGD step
            server_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            server_optimizer.step()

        if (step + 1) % args.print_freq == 0:
            print("\tServer Iter [%d/%d] Loss: %.4f" % (step + 1, train_size, loss.item()))


# Validation and Testing
def eval(models, data_loader):
    """
    Calculate loss and accuracy for a given data_loader
    """
    total = 0.0
    correct = 0.0

    total_loss = 0.0
    n = 0

    for i, (inputs, targets) in enumerate(data_loader):
        with torch.no_grad():
            # Convert from list of 3D to 4D
            inputs = np.stack(inputs, axis=1)

            inputs = torch.from_numpy(inputs)

            inputs, targets = inputs.cuda(device), targets.cuda(device)
            inputs, targets = Variable(inputs), Variable(targets)

            # Get current embeddings
            H_new = [None] * num_clients
            for i in range(num_clients):
                r = math.floor(i/2)
                c = i % 2
                section = coords_per
                # print(inputs.shape)
                # print(section*r, section*(r+1), section*c, section*(c+1))
                x_local = inputs[:, section*r:section*(r+1)]
                x_local = torch.transpose(x_local,0,1)
                H_new[i] = models[i](x_local)
            # compute output
            outputs = models[-1](torch.cat(H_new,axis=1))

            # print(outputs.shape, targets.shape)
            loss = criterion(outputs, targets.to(torch.int64))

            total_loss += loss
            n += 1

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted.cpu() == targets.cpu()).sum()

    avg_test_acc = 100 * correct / total
    avg_loss = total_loss / n

    return avg_test_acc, avg_loss

print('[111]')
# Get initial loss/accuracy
if start_epoch == 0:
    save_eval(models, train_loader, test_loader, losses, accs_train, accs_test, 0, len(train_loader))
# Training / Eval loop
train_size = len(train_loader)
print('[222]')
assert start_epoch == 0
for epoch in range(n_epochs):
    print('\n-----------------------------------')
    print('Epoch: [%d/%d]' % (epoch+1, n_epochs))
    start = time.time()

    train(models, optimizers, epoch)
    save_eval(models, train_loader, test_loader, losses, accs_train, accs_test, epoch, train_size)

    for i in range(num_clients+1):
        PATH = f"./cp{i}{suffix}_{epoch}.pt"

        torch.save({
            'model': models[i].state_dict(),
        }, PATH)
    print('Time taken: %.2f sec.' % (time.time() - start))
