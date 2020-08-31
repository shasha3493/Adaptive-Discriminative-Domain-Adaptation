# Importing Relevant Libraries 

from logging import getLogger
from time import time
import numpy as np
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter
import torch
from utils import AverageMeter, save


logger = getLogger('adda.trainer')


def train_source_cnn(
    source_cnn, train_loader, test_loader, criterion, optimizer,
    args=None
):
    '''
    Train the source classification network
    '''
    best_score = None
    for epoch_i in range(1, 1 + args.epochs):
        
        # is dictionary containing avg loss/sample and accuracy across the entire training dataset
        training = train(
            source_cnn, train_loader, criterion, optimizer, args=args)

        # is a dictionary containing avg loss/sample and accuracy across the entire training dataset
        validation = validate(
            source_cnn, test_loader, criterion, args=args)

        print('Epoch {}/{} | Train Loss: {} | Train Acc: {} | Val Loss: {} | Val Acc: {}'.format(epoch_i, args.epochs, training['loss'], training['acc'], validation['loss'], validation['acc']))
        
        # saving the model if validation accuarcy is better than best_score
        is_best = (best_score is None or validation['acc'] > best_score)
        best_score = validation['acc'] if is_best else best_score
        state_dict = {
            'model': source_cnn.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch_i,
            'val/acc': best_score,
        }
        save(args.logdir, state_dict, is_best)

    return source_cnn


def train_target_cnn(
    source_cnn, target_cnn, discriminator,
    criterion, optimizer, d_optimizer,
    source_train_loader, target_train_loader, target_test_loader,
    args=None
):
    # Checking the accuracy on target domain images by source cnn for comparison
    # dictionary of avg_loss/sample, accuracy of target_test_loader
    validation = validate(source_cnn, target_test_loader, criterion, args=args)
    print('Without Domain Adaptation, accuracy of classfication network on target domain\'s test data: {}'.format(validation['acc']))
    print('################################################################################################################')
    
    best_score = None

    # Training target cnn and discriminator
    for epoch_i in range(1, 1 + args.epochs):

        # One epoch of training of target classification network's encoder and discriminator
        # a dictionary containing containing avg loss/batch for discriminator and target cnn encoder
        training = adversarial(
            source_cnn, target_cnn, discriminator,
            source_train_loader, target_train_loader,
            criterion, criterion,
            optimizer, d_optimizer,
            args=args
        )

        # a dictionary of avg_loss/sample and accuracy of target classification network on 
        # test data of target domain
        validation = validate(
            target_cnn, target_test_loader, criterion, args=args)

        # a dictionary of avg_loss/sample and accuracy of target classification network 
        # on train data of target domain
        validation2 = validate(
            target_cnn, target_train_loader, criterion, args=args)

        print('Epoch {}/{} | D Loss: {} | Target encoder loss: {} | Train Loss: {} | Train Acc: {} \
              | Val Loss: {} | Val Acc: {}'.format(epoch_i, args.epochs, training['d/loss'], 
              training['target/loss'], validation['loss'], validation['acc'], validation2['loss'], validation2['acc']))
        
        # save the model if test accuracy of target cnn on test images of target domain is better
        is_best = (best_score is None or validation['acc'] > best_score)
        best_score = validation['acc'] if is_best else best_score
        state_dict = {
            'model': target_cnn.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch_i,
            'val/acc': best_score,
        }
        save(args.logdir, state_dict, is_best)


    target_cnn.load_state_dict(torch.load(args.logdir))
    validation = validate(target_cnn, target_test_loader, criterion, args=args)
    print('After Domain Adaptation, accuracy of classfication network on target domain\'s test data: {}'.format(validation['acc']))
    print('################################################################################################################')

def adversarial(
    source_cnn, target_cnn, discriminator,
    source_loader, target_loader,
    criterion, d_criterion,
    optimizer, d_optimizer,
    args=None
):

    '''
    Does training of target classification network's encoder, and discriminator for one epoch. 
    Source cnn is not trained and hence kept in eval mode

    Returns a dictionary containing avg loss/sample for discriminator and target classification
    network's encoder

    '''
    source_cnn.eval()
    target_cnn.encoder.train()
    discriminator.train()

    losses, d_losses = AverageMeter(), AverageMeter()

    # Consider the minimum of source and target loader
    n_iters = min(len(source_loader), len(target_loader))

    # Creating iterator from data loader
    source_iter, target_iter = iter(source_loader), iter(target_loader)

    # Going through every batch to cover the entire dataset once
    for iter_i in range(n_iters):
        source_data, source_target = source_iter.next()
        target_data, target_target = target_iter.next()
        source_data = source_data.to(args.device)
        target_data = target_data.to(args.device)
        bs = source_data.size(0)

        # Output of source classification network's encoder with source domain images as input
        D_input_source = source_cnn.encoder(source_data)

        # Output of target classification network's encoder with target domain images as input
        D_input_target = target_cnn.encoder(target_data)

        # Creating true labels for the source and target domain images to be used for discriminator
        
        #tensor of shape [batch_size] containing all zeros to indicate that the images are from
        # source domain 
        D_target_source = torch.tensor(
            [0] * bs, dtype=torch.long).to(args.device)

        #tensor of shape [batch_size] containing all zeros to indicate that the images are from
        # target domain 
        D_target_target = torch.tensor(
            [1] * bs, dtype=torch.long).to(args.device)

        ###################################
            #  Train Discriminator
        ###################################

        # Output of discriminator on the source domain images
        D_output_source = discriminator(D_input_source) #(bs, 2)

        # Output of discriminator on the target domain images
        D_output_target = discriminator(D_input_target) #(bs, 2)

        # Combining the output of discriminator on the source and target domain images
        D_output = torch.cat([D_output_source, D_output_target], dim=0) #(2*bs,2)

        # Combining the true label
        D_target = torch.cat([D_target_source, D_target_target], dim=0) #(2*bs, )

        d_loss = criterion(D_output, D_target)
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        d_losses.update(d_loss.item(), bs)

        ##############################################################
            #  Train Target Classification Network's Encoder
        ##############################################################

        D_input_target = target_cnn.encoder(target_data)
        D_output_target = discriminator(D_input_target)

        # True label given is 0 as it wasnts to trick the discriminator
        loss = criterion(D_output_target, D_target_source)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), bs)

    return {'d/loss': d_losses.avg, 'target/loss': losses.avg}


def step(model, data, target, criterion, args):
    '''
    Accepts model, one batch of data and target, criterion function
    Returns output and loss for the batch
    '''

    # Sending data, target to device
    data, target = data.to(args.device), target.to(args.device) #(batch_size,3,32,32), (bs, )
    
    # output of the model
    output = model(data) # (bs, 10)

    # Calculating loss for the batch
    loss = criterion(output, target)
    return output, loss


def train(model, dataloader, criterion, optimizer, args=None):
    '''
    One Training epoch

    Accepts model, dataloader, criterion, optimizer 
    Returns Avg loss/batch for one epoch and accuracy across the entire taraining dataset
    '''

    model.train()
    losses = AverageMeter()
    targets, probas = [], []
    for i, (data, target) in enumerate(dataloader):
        
        # data: (batch_size, 3, 32, 32)
        # target: (batch_size)

        # Goes through every batch in the dataset
        bs = target.size(0) # batch size

        output, loss = step(model, data, target, criterion, args)
        
        # model's score -> probabilities
        output = torch.softmax(output, dim=1)  # (bs, 10)
        
        # Updating avg loss/batch
        losses.update(loss.item(), bs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Storing true labels for the batch and extending 
        targets.extend(target.cpu().detach().numpy().tolist())
        # Storing model predicted label for the batch and extending 
        probas.extend(output.cpu().detach().numpy().tolist())
    probas = np.asarray(probas)
    preds = np.argmax(probas, axis=1)

    # Finding accuracy for the entire training dataset
    acc = accuracy_score(targets, preds)
    
    # Returning avg loss/sample and accuracy along the entire training dataset
    return {
        'loss': losses.avg, 'acc': acc,
    }


# Same as train
def validate(model, dataloader, criterion, args=None):
    '''
    One validation cycle

    Accepts model, dataloader, criterion
    Returns avg_loss/sample and accuracy across the entire valdation dataset
    '''
    # Putting the model in eval mode
    model.eval()
    losses = AverageMeter()
    targets, probas = [], []
    with torch.no_grad():

        for iter_i, (data, target) in enumerate(dataloader): # data: (bs, 3, 32, 32) # target: (bs,)
            
            bs = target.size(0) # output: (bs, 10)
            output, loss = step(model, data, target, criterion, args)
            output = torch.softmax(output, dim=1)  # Probabilities
            losses.update(loss.item(), bs)
            targets.extend(target.cpu().numpy().tolist())
            probas.extend(output.cpu().numpy().tolist())
    probas = np.asarray(probas)
    preds = np.argmax(probas, axis=1) # Predicted Labels
    acc = accuracy_score(targets, preds) # Accuracy
    return {
        'loss': losses.avg, 'acc': acc,
    }
