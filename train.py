#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#

import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F

from tqdm import tqdm

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from modules.optim import ScheduledOptimizer
from modules.early_stopping import EarlyStopping

from vocab import Vocab
from sa_model import SAModel
from dataset import load_data, build_dataloader

# Parse argument for language to train
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='')
parser.add_argument('--data_dir', type=str, help='')
parser.add_argument('--vocab_path', type=str, help='')
parser.add_argument('--vocab_size', type=int, help='')
parser.add_argument('--embedding_size', type=int)
parser.add_argument('--hidden_size', type=int)
parser.add_argument('--bidirectional', action='store_true')
parser.add_argument('--num_layers', type=int)
parser.add_argument('--transformer_size', type=int)
parser.add_argument('--inner_hidden_size', type=int)
parser.add_argument('--dense_size', type=int)
parser.add_argument('--use_pretrained_embeddings', action='store_ture')
parser.add_argument('--n_classes', type=int)
parser.add_argument('--t_num_layers', type=int)
parser.add_argument('--k_size', type=int)
parser.add_argument('--v_size', type=int)
parser.add_argument('--num_heads', type=int)
parser.add_argument('--dropout', type=float)
parser.add_argument('--max_grad_norm', type=float, default=5.0)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--min_len', type=int, default=5)
parser.add_argument('--max_len', type=int, default=60)
parser.add_argument('--batch_size', type=int, help='')
parser.add_argument('--valid_split', type=float, default=0.08)
parser.add_argument('--test_split', type=int, default=5)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--start_epoch', type=int, default=1)
parser.add_argument('--lr_patience', type=int,
                    help='Number of epochs with no improvement after which learning rate will be reduced')
parser.add_argument('--es_patience', type=int, help='early stopping patience.')
parser.add_argument('--device', type=str, help='cpu or cuda')
parser.add_argument('--save_model', type=str, help='save path')
parser.add_argument('--save_mode', type=str,
                    choices=['all', 'best'], default='best')
parser.add_argument('--checkpoint', type=str, help='checkpoint path')
parser.add_argument('--smoothing', action='store_true')
parser.add_argument('--log', type=str, help='save log.')
parser.add_argument('--seed', type=str, help='random seed')
parser.add_argument('--model_type', type=str, help='')
parser.add_argument('--mode', type=str, help='train, eval, infer')
args = parser.parse_args()

print(' '.join(sys.argv))

torch.random.manual_seed(args.seed)
device = torch.device(args.device)
print('device: {}'.format(device))

# load vocab
vocab = Vocab()
vocab.load(args.vocab_path)
args.vocab_size = int(vocab.size)
print('vocab size: ', args.vocab_size)

# load data
datas = load_data(args, vocab)

# dataset, data_load
train_data, valid_data, test_data = build_dataloader(args, datas)

# model
model = SAModel(
    args,
    device
).to(device)

print(model)

# optimizer
#  optimizer = optim.Adam(model.parameters(), lr=args.lr)
optim = torch.optim.Adam(
    model.parameters(),
    args.lr,
    betas=(0.9, 0.98),
    eps=1e-09
)

#  scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=2, gamma=0.5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optim,
    mode='min',
    factor=0.1,
    patience=args.lr_patience
)

optimizer = ScheduledOptimizer(
    optim,
    scheduler,
    args.max_grad_norm
)

# early stopping
early_stopping = EarlyStopping(
    type='min',
    min_delta=0.001,
    patience=args.es_patience
)

# train epochs

def train_epochs():
    ''' Start training '''
    log_train_file = None
    log_valid_file = None

    if args.log:
        log_train_file = os.path.join(args.log, 'train.log')
        log_valid_file = os.path.join(args.log, 'valid.log')

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, \
                open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch, loss, accuracy, recall, f1\n')
            log_vf.write('epoch, loss, accuracy, recall, f1\n')

    valid_accuracies = []
    for epoch in range(args.start_epoch, args.epochs + 1):
        print('[ Epoch', epoch, ']')

        start = time.time()

        train_loss, train_accuracy, train_recall, train_f1 = train(epoch)

        print(' (Training) loss: {loss: 8.5f}, accuracy: {accuracy:3.3f} %, '
              'recall: {recall:3.3f} %, f1: {f1: 3.3f} %'
              'elapse: {elapse:3.3f} min'.format(
                  loss=train_loss,
                  accuracy=100*train_accuracy,
                  recall=100*train_recall,
                  f1=100*train_f1,
                  elapse=(time.time()-start)/60)
              )

        start = time.time()
        valid_loss, valid_accuracy, valid_recall, valid_f1 = eval(epoch)
        print(' (Validation) loss: {ppl: 8.5f}, accuracy: {accuracy:3.3f} %, '
              'recall: {accuracy:3.3f} %, f1: {f1: 3.3f} %'
              'elapse: {elapse:3.3f} min'.format(
                  loss=valid_loss,
                  accuracy=100*valid_accuracy,
                  recall=100*valid_recall,
                  f1=100*valid_f1,
                  elapse=(time.time()-start)/60)
              )

        valid_accuracies += [valid_accuracy]

        # is early_stopping
        is_stop = early_stopping.step(valid_loss)

        checkpoint = {
            'model': model.state_dict(),
            'settings': args,
            'epoch': epoch,
            'optimizer': optimizer.optimizer.state_dict(),
            'valid_loss': valid_loss,
            'valid_accuracy': valid_accuracy,
            'valid_recall': valid_recall,
            'valid_f1': valid_f1
        }

        if args.save_model:
            if args.save_mode == 'all':
                model_name = os.path.join(
                    args.save_model,
                    'accuracy_{accuracy:3.3f}.pth'.format(accuracy=100*valid_accuracy)
                )
                torch.save(checkpoint, model_name)
            elif args.save_mode == 'best':
                model_name = os.path.join(args.save_model, 'best.pth')
                if valid_accuracy >= max(valid_accuracies):
                    torch.save(checkpoint, model_name)
                    print('   - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch}, {loss: 8.5f}, {accuracy:3.3f}, {recall:3.3f}, {f1:3.3f}\n'.format(
                    epoch=epoch,
                    loss=train_loss,
                    accuracy=100*train_accuracy,
                    recall=100*train_recall,
                    f1=100*train_f1)
                )
                log_vf.write('{epoch}, {loss: 8.5f}, {accuracy:3.3f}, {recall: 3.3f}, {f1:3.3f}\n'.format(
                    epoch=epoch,
                    loss=valid_loss,
                    accuracy=100*valid_accuracy,
                    recall=100*valid_recall,
                    f1=100*valid_f1)
                )

        if is_stop:
            print('Early Stopping.\n')
            sys.exit(0)

# train

def train(epoch):
    ''' Epoch operation in training phase'''
    model.train()

    total_loss = 0
    total_accuracy = 0
    total_recall = 0
    total_f1 = 0
    times = 0

    for batch in tqdm(
            train_data, mininterval=2,
            desc=' (Training: %d) ' % epoch, leave=False):

        # prepare data
        inputs, lengths, labels = map(lambda x: x.to(device), batch)
        # [batch_size, max_len]

        # forward
        optimizer.zero_grad()

        outputs, attns = model(
            inputs,
            lengths
        )

        # backward
        loss, accuracy, recall, f1 = cal_performance(
            outputs,
            labels,
            smoothing=args.smoothing
        )

        loss.backward()

        # update parameters
        optimizer.step()

        # note keeping
        total_loss += loss.item()
        total_accuracy += accuracy
        total_recall += recall
        total_f1 += f1
        times += 1

    avg_loss = total_loss / times
    avg_accuracy = total_accuracy / times
    avg_recall = total_recall / times
    avg_f1 = total_f1 / times

    return avg_loss, avg_accuracy, avg_recall, avg_f1


def eval(epoch):
    ''' Epoch operation in evaluation phase '''
    model.eval()

    total_loss = 0
    total_accuracy = 0
    total_recall = 0
    total_f1 = 0
    times = 0

    with torch.no_grad():
        for batch in tqdm(
                valid_data, mininterval=2,
                desc=' (Validation: %d) ' % epoch, leave=False):

            inputs, lengths, labels = map(lambda x: x.to(device), batch)

            outputs = model(
                inputs,
                lengths
            )

            # backward
            loss, accuracy, recall, f1 = cal_performance(
                outputs,
                labels,
                smoothing=args.smoothing
            )

            # note keeping
            total_loss += loss.item()
            total_accuracy += accuracy
            total_recall += recall
            total_f1 += f1
            times += 1

    avg_loss = total_loss / times
    avg_accuracy = total_accuracy / times
    avg_recall = total_recall / times
    avg_f1 = total_f1 / times

    return avg_loss, avg_accuracy, avg_recall, avg_f1

def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''
    # pred: [batch_size, n_classes]
    # gold: [batch_size]

    loss = cal_loss(pred, gold, smoothing)

    # [batch_size]
    pred = pred.max(dim=1)[1]

    # [batch_size]
    gold = gold.contiguous().view(-1)

    accuracy = accuracy_score(gold.tolist(), pred.tolist())
    recall = recall_score(gold.tolist(), pred.tolist(), average='micro')
    f1 = f1_score(gold.tolist(), pred.tolist(), average='micro')

    return loss, accuracy, recall, f1


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    # [max_len * batch_size]
    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_classes = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_classes - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, reduction='sum')

    return loss


if __name__ == '__main__':
    mode = args.mode

    if args.checkpoint:
        print('load checkpoint...')
        checkpoint = torch.load(args.checkpoint)

        model.load_state_dict(checkpoint['model'])
        optimizer.optimizer.load_state_dict(checkpoint['optimizer'])

        #  early_stopping = checkpoint['early_stopping']

        args = checkpoint['settings']

        epoch = checkpoint['epoch']
        args.start_epoch = epoch + 1

        valid_loss = checkpoint['valid_loss']
        valid_accuracy = checkpoint['valid_accuracy']
        valid_recall = checkpoint['valid_recall']
        valid_f1 = checkpoint['valid_f1']

        print(
            '  - (checkpoint) epoch: {epoch: d}, loss: {loss: 8.5f}, '
            'accuracy: {accuracy:3.3f}%, recall: {recall:3.3f}%, f1: {f1:3.3f}%'.format(
                epoch=epoch,
                loss=valid_loss,
                accuracy=100*valid_accuracy,
                recall=100*valid_recall,
                f1=100*valid_f1
            )
        )

    args.mode = mode

    if args.mode == 'train':
        train_epochs()
    elif args.mode == 'eval':
        eval(epoch)
