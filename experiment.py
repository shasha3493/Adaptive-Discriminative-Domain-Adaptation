# Importing Relevant Libraries

import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN, MNIST
from torchvision import transforms
from models import CNN, Discriminator
from trainer import train_target_cnn
from utils import get_logger


def run(args):

    # Creating directory (if it doesn't exist) to store model
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    # Souce domain Image tranformation
        # Converting image to tensor
    source_transform = transforms.Compose([
        # transforms.Grayscale(),
        transforms.ToTensor()]
    )

    # target domain Image transformation
        # Resizing to 32*32
        # Converting image to tensor
        # Handling grayscale images
    target_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)) # if the image is gray scale, 
                                                       # copy the channel thrice
    ])

    # Source domain train dataset
    source_dataset_train = SVHN(
        './input', 'train', transform=source_transform, download=True)
    
    # target domain train dataset
    target_dataset_train = MNIST(
        './input', 'train', transform=target_transform, download=True)
    
    # target domain train dataset
    target_dataset_test = MNIST(
        './input', 'test', transform=target_transform, download=True)
    
    # source domain train loader
    source_train_loader = DataLoader(
        source_dataset_train, args.batch_size, shuffle=True,
        drop_last=True,
        num_workers=args.n_workers)
    
    # target domain train loader
    target_train_loader = DataLoader(
        target_dataset_train, args.batch_size, shuffle=True,
        drop_last=True,
        num_workers=args.n_workers)
    
    # target domain test loader
    target_test_loader = DataLoader(
        target_dataset_test, args.batch_size, shuffle=False,
        num_workers=args.n_workers)
      
    
    # Source Classification Network
    # target = false as we need to train the classifier also
    source_cnn = CNN(in_channels=args.in_channels).to(args.device)
    
    # load the model if checkpoint exists for the source classification architecture
    if os.path.isfile(args.trained):
        
        c = torch.load(args.trained)
        source_cnn.load_state_dict(c['model'])
        # logger.info('Loaded `{}`'.format(args.trained))
    
    # Target Classification architecture
    # target = true as we needn't train the classifier as classifier for both source 
    # and target classification network is same 
    target_cnn = CNN(in_channels=args.in_channels, target=True).to(args.device)

    # Initialize the weights with the trained weights of source classification network
    target_cnn.load_state_dict(source_cnn.state_dict())

    # Build the discriminator architecture and initializes the weights
    discriminator = Discriminator(args=args).to(args.device)

    criterion = nn.CrossEntropyLoss()

    # Optimizer for encoder of source and target classification network
    optimizer = optim.Adam(
        target_cnn.encoder.parameters(),
        lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)

    # Optimizer for discriminator
    d_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)

    # Fine Tune the encoder network of target classification network
    train_target_cnn(
        source_cnn, target_cnn, discriminator,
        criterion, optimizer, d_optimizer,
        source_train_loader, target_train_loader, target_test_loader,
        args=args)
