#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#

import os
import sys
import time
import argparse

# import warnings
# warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

import torch
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
#  from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

from modules.optim import ScheduledOptimizer
from modules.early_stopping import EarlyStopping

from sa_model import SAModel

from misc.vocab import Vocab
from misc.vocab import PAD_ID

from misc.dataset import load_data, build_dataloader
from misc.tokenizer import Tokenizer

from visualization.self_attention.visualization import create_html
from visualization.transformer import seaborn_draw

# Parse argument for language to train
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='')
parser.add_argument('--data_dir', type=str, help='')
parser.add_argument('--visualization_dir', type=str, help='')
parser.add_argument('--vocab_path', type=str, help='')
parser.add_argument('--vocab_size', type=int, help='')
parser.add_argument('--rnn_type', type=str, help='RNN, LSTM, GRU')
parser.add_argument('--embedding_size', type=int)
parser.add_argument('--hidden_size', type=int)
parser.add_argument('--bidirectional', action='store_true')
parser.add_argument('--use_pos', action='store_true')
parser.add_argument('--sampler', action='store_true')
parser.add_argument('--num_layers', type=int)
parser.add_argument('--transformer_size', type=int)
parser.add_argument('--inner_hidden_size', type=int)
parser.add_argument('--dense_size', type=int)
parser.add_argument('--regression_dense_size', type=int)
parser.add_argument('--use_pretrained_embedding', action='store_true')
parser.add_argument('--n_classes', type=int)
parser.add_argument('--t_num_layers', type=int)
parser.add_argument('--k_size', type=int)
parser.add_argument('--v_size', type=int)
parser.add_argument('--num_heads', type=int)
parser.add_argument('--in_channels', type=int)
parser.add_argument('--out_channels', type=int)
# https://stackoverflow.com/questions/15753701/argparse-option-for-passing-a-list-as-option/15753721#15753721
parser.add_argument('--kernel_heights', nargs='+', type=int, help='')
parser.add_argument('--stride', type=int)
parser.add_argument('--padding', type=int)
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
parser.add_argument('--use_penalization', action='store_true')
parser.add_argument('--penalization_coeff', type=float, default=0.00)
parser.add_argument('--log', type=str, help='save log.')
parser.add_argument('--log_mode', type=str, default='w', help='w or a')
parser.add_argument('--seed', type=str, help='random seed')
parser.add_argument('--model_type', type=str, help='')
parser.add_argument('--problem', type=str, help='classification or regression')
parser.add_argument('--mode', type=str, help='train, eval, test')
parser.add_argument('--text', type=str, default='', help='text for testing.')
parser.add_argument('--classes_weight', nargs='+', type=float, help='')
parser.add_argument('--classes_count', nargs='+', type=float, help='')
parser.add_argument('-n_warmup_steps', type=int, default=4000)
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
train_data, valid_data, test_data, args = build_dataloader(args, datas)
args.classes_weight = args.classes_weight.to(device)
print('train classes_count: {}'.format(args.classes_count))
print('train classes_weight: {}'.format(args.classes_weight))

# load pretrained embedding TODO
pretrained_embedding = None

# model
model = SAModel(
    args,
    device,
    pretrained_embedding
).to(device)

print(model)

# optimizer
#  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
optim = torch.optim.Adam(
    filter(lambda x: x.requires_grad, model.parameters()),
    #  args.lr,
    betas=(0.9, 0.98),
    eps=1e-09
)

optimizer = ScheduledOptimizer(
    optim,
    args.embedding_size,
    args.n_warmup_steps
)


"""
#  scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=2, gamma=0.5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optim,
    mode='min',
    factor=0.1,
    patience=args.lr_patience
)

"""
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
        log_train_file = os.path.join(args.log, 'train.%s.%s.log' % (args.problem, args.model_type))
        log_valid_file = os.path.join(args.log, 'valid.%s.%s.log' % (args.problem, args.model_type))

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        if args.log_mode == 'w':
            with open(log_train_file, 'w') as log_tf, \
                    open(log_valid_file, 'w') as log_vf:
                if args.problem == 'classification':
                    log_tf.write('epoch, loss, accuracy, recall, f1\n')
                    log_vf.write('epoch, loss, accuracy, recall, f1\n')
                else:
                    log_tf.write('epoch, loss, \n')
                    log_vf.write('epoch, loss, \n')

    valid_accuracies = []
    for epoch in range(args.start_epoch, args.epochs + 1):
        print('[ Epoch', epoch, ']')

        start = time.time()
        if args.problem == 'classification':
            train_loss, train_accuracy, train_recall, train_f1 = train(epoch)
            print(' (Training) loss: {loss: 8.5f}, accuracy: {accuracy:3.3f}%, '
                  'recall: {recall:3.3f}%, f1: {f1: 3.3f}%, '
                  'elapse: {elapse:3.3f}min'.format(
                      loss=train_loss,
                      accuracy=100*train_accuracy,
                      recall=100*train_recall,
                      f1=100*train_f1,
                      elapse=(time.time()-start)/60)
                  )
        else:
            train_loss = train(epoch)
            print(' (Training) loss: {loss: 8.5f}, '
                  'elapse: {elapse:3.3f}min'.format(
                      loss=train_loss,
                      elapse=(time.time()-start)/60)
                  )

        start = time.time()
        if args.problem == 'classification':
            valid_loss, valid_accuracy, valid_recall, valid_f1 = eval(epoch)
            print(' (Validation) loss: {loss: 8.5f}, accuracy: {accuracy:3.3f}%, '
                  'recall: {accuracy:3.3f}%, f1: {f1: 3.3f}%, '
                  'elapse: {elapse:3.3f} min'.format(
                      loss=valid_loss,
                      accuracy=100*valid_accuracy,
                      recall=100*valid_recall,
                      f1=100*valid_f1,
                      elapse=(time.time()-start)/60)
                  )
            valid_accuracies += [valid_accuracy]
        else:
            valid_loss = eval(epoch)
            print(' (Validataion) loss: {loss: 8.5f}, '
                  'elapse: {elapse:3.3f}min'.format(
                      loss=valid_loss,
                      elapse=(time.time()-start)/60)
                  )

        # is early_stopping
        is_stop = early_stopping.step(valid_loss)

        checkpoint = {
            'model': model.state_dict(),
            'settings': args,
            'epoch': epoch,
            'optimizer': optimizer._optimizer.state_dict(),
            'valid_loss': valid_loss,
        }
        if args.problem == 'classification':
            checkpoint['valid_accuracy'] = valid_accuracy
            checkpoint['valid_recall'] = valid_recall
            checkpoint['valid_f1'] = valid_f1

        if args.save_model:
            if args.save_mode == 'all':
                if args.problem == 'classification':
                    model_name = os.path.join(
                        args.save_model,
                        'classification.{}.accuracy_{accuracy:3.3f}.pth'.format(
                            args.model_type,
                            accuracy=100*valid_accuracy))
                else:
                    model_name = os.path.join(
                        args.save_model,
                        'regression.{}.loss_{loss:6.5f}.pth'.format(
                            args.model_type,
                            loss=valid_loss))

                torch.save(checkpoint, model_name)
            elif args.save_mode == 'best':
                if args.problem == 'classification':
                    model_name = os.path.join(
                        args.save_model, 'classification.%s.best.pth' % args.model_type)
                else:
                    model_name = os.path.join(
                        args.save_model, 'regression.%s.best.pth' % args.model_type)
                if valid_accuracy >= max(valid_accuracies):
                    torch.save(checkpoint, model_name)
                    print('   - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                if args.problem == 'classification':
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
                else:
                    log_tf.write('{epoch}, {loss: 8.5f}, \n'.format(
                        epoch=epoch,
                        loss=train_loss,))
                    log_vf.write('{epoch}, {loss: 8.5f}, \n'.format(
                        epoch=epoch,
                        loss=valid_loss,))

        if is_stop:
            print('Early Stopping.\n')
            sys.exit(0)

def train(epoch):
    ''' Epoch operation in training phase'''
    model.train()

    total_loss = 0
    total_label = 0
    times = 0
    if args.problem == 'classification':
        total_accuracy = 0
        total_recall = 0
        total_f1 = 0

    for batch in tqdm(
            train_data, mininterval=2,
            desc=' (Training: %d) ' % epoch, leave=False):

        # prepare data
        inputs, lengths, labels, inputs_pos = map(
            lambda x: x.to(device), batch)
        # [batch_size, max_len]
        # print('inputs: ', inputs)
        # print('legnths: ', lengths)
        # print('labels: ', labels)

        # forward
        optimizer.zero_grad()

        loss = 0

        outputs, attns = model(
            inputs,
            lengths=lengths,
            inputs_pos=inputs_pos
        )

        if args.problem == 'classification':
            # self attention, penalization AA - I
            if args.model_type == 'self_attention' and args.use_penalization:
                loss, accuracy, recall, f1 = cal_performance(outputs.double() + 1e-8, labels)

                # [bath_size, max_len, num_heads]
                attnsT = attns.transpose(1, 2)
                # [num_heads, num_heads]
                identity = torch.eye(attns.size(1), device=device)
                # [batch_size, num_heads, num_heads]
                identity = identity.unsqueeze(0).expand(attns.size(0), attns.size(1), attns.size(1))
                # print('attns: ', attns.shape)
                # print('attnsT: ', attnsT.shape)
                # print('identity: ', identity.shape)

                penalization = model.encoder.l2_matrix_norm(attns @ attnsT - identity)

                loss = loss + args.penalization_coeff * penalization / args.batch_size
                #  loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1)+1e-8,y)
                #  + C * penal/train_loader.batch_size
            else:
                loss, accuracy, recall, f1 = cal_performance(outputs.double(), labels)

        else:
            loss = cal_performance(outputs.double(), labels.double())

        # backward
        loss.backward()

        # update parameters
        optimizer.step_and_update_lr()

        # note keeping
        total_loss += loss.item()
        times += 1
        if args.problem == 'classification':
            total_label += labels.size(0)
            total_accuracy += accuracy
            total_recall += recall
            total_f1 += f1

    if args.problem == 'classification':
        avg_loss = total_loss / total_label
        avg_accuracy = total_accuracy / times
        avg_recall = total_recall / times
        avg_f1 = total_f1 / times
        return avg_loss, avg_accuracy, avg_recall, avg_f1
    else:
        avg_loss = total_loss / times
        return avg_loss


def eval(epoch):
    ''' Epoch operation in evaluation phase '''
    model.eval()

    total_loss = 0
    total_label = 0
    times = 0
    if args.problem == 'classification':
        total_accuracy = 0
        total_recall = 0
        total_f1 = 0

    with torch.no_grad():
        for batch in tqdm(
                valid_data, mininterval=2,
                desc=' (Validation: %d) ' % epoch, leave=False):

            inputs, lengths, labels, inputs_pos = map(
                lambda x: x.to(device), batch)
            # print('inputs: ', inputs)
            # print('legnths: ', lengths)
            # print('labels: ', labels)

            outputs, attns = model(
                inputs,
                lengths=lengths,
                inputs_pos=inputs_pos
            )

            if args.problem == 'classification':
                loss, accuracy, recall, f1 = cal_performance(outputs.double(), labels)
            else:
                loss = cal_performance(outputs.double(), labels.double())

            # note keeping
            total_loss += loss.item()

            times += 1
            if args.problem == 'classification':
                total_label += labels.size(0)
                total_accuracy += accuracy
                total_recall += recall
                total_f1 += f1

    if args.problem == 'classification':
        avg_loss = total_loss / total_label
        avg_accuracy = total_accuracy / times
        avg_recall = total_recall / times
        avg_f1 = total_f1 / times
        return avg_loss, avg_accuracy, avg_recall, avg_f1
    else:
        avg_loss = total_loss / times
        return avg_loss


def test():
    if args.text is None or len(args.text.split()) == 0:
        raise ValueError('text: %s is invalid' % args.text)

    tokenizer = Tokenizer()
    with torch.no_grad():
        tokens = tokenizer.tokenize(args.text)
        ids = vocab.words_to_id(tokens)
        ids = ids + [PAD_ID] * (args.max_len - len(ids))  # pad
        pos = [pos_i + 1 if w_i !=
               PAD_ID else 0 for pos_i, w_i in enumerate(ids)]

        input = torch.LongTensor(ids)
        input = input.to(device)

        input_pos = torch.LongTensor(pos)
        input_pos = input_pos.to(device)

        # unsqueeze batch_size
        inputs = input.unsqueeze(1)  # [max_len, 1]
        inputs_pos = input_pos.unsqueeze(1)  # [max_len, 1]

        # print('input: ', input)

        # outputs: [1, n_classes], [num_heads * 1, max_len, max_len] list or [1, num_heads, max_len]
        outputs, attns = model(
            inputs=inputs,
            inputs_pos=inputs_pos
        )

        if args.problem == 'classification':
            outputs = F.log_softmax(outputs, dim=1)
            print('outputs: ', outputs)

            label = outputs.squeeze(0).topk(1)[1].item()
            # print('attns: ', attns.shape)

            # print('len(attns): ', len(attns))
            # print('attns[0]: ', attns[0].shape)
            # print('attns[-1]: ', attns[-1].shape)

            print('text: %s, label: %d' % (args.text, label))
            # print('attns: ', attns)

            # tokens = ['x'] * len(tokens)
            # print('tokens: ', tokens)

            # visualize_self_attention(attns, ids)
            # visualize_transformer(attns, tokens)
        else:
            # outputs: [1, 1]
            print('outputs: ', outputs)
            print('text: %s, score: %f' % (args.text, outputs.item()))


def visualize_self_attention(attns, ids):
    attns_add = torch.sum(attns, 1)  # [batch_size, max_len]
    attns_add_np = attns_add.data.cpu().numpy()
    attns_add_list = attns_add_np.tolist()

    texts = []
    tokens = vocab.ids_to_word(ids)
    texts.append(' '.join(tokens))

    create_html(
        texts=texts,
        weights=attns_add_list,
        file_name=os.path.join(args.visualization_dir,
                               args.text.replace(' ', '') + '.html')
    )
    print("Attention visualization created for {} samples".format(len(texts)))


def visualize_transformer(attns, tokens):
    for layer in range(args.t_num_layers):
        fig, axs = plt.subplots(1, args.num_heads, figsize=(20, 10))
        # cbar_ax = fig.add_axes([.905, .3, .05, .3])
        print("self attn Layer: ", layer + 1)
        for h in range(args.num_heads):
            seaborn_draw.draw(
                attns[layer][h].data.cpu().numpy()[:len(tokens), :len(tokens)],
                tokens,
                tokens if h == 0 else [],
                ax=axs[h],
                # cbar_ax=cbar_ax if h == args.num_heads - 1 else None,
                # cbar=True if h == args.num_heads - 1 else False
            )
    # plt.show()
    plt.savefig(os.path.join(args.visualization_dir,
                             args.text.replace(' ', '') + '.png'))


def cal_performance(pred, gold):
    ''' Apply label smoothing if needed '''
    # pred: [batch_size, n_classes]
    # gold: [batch_size]
    # print('pred shape: ', pred.shape)
    # print('pred : {}'.format(pred))
    # print('gold shape: ', gold.shape)
    # print('gold : {}'.format(gold))

    loss = cal_loss(pred, gold, args.smoothing)

    if args.problem == 'classification':
        # [batch_size]
        pred = pred.max(dim=1)[1].tolist()

        # [batch_size]
        gold = gold.contiguous().view(-1).tolist()

        accuracy = accuracy_score(gold, pred)

        def intersection(list1, list2):
            # Use of hybrid method
            temp1 = set(list1)
            temp2 = set(list2)
            list3 = [value for value in temp1 if value in temp2]
            return list3

        labels = intersection(gold, pred)
        # print('labels: ', labels)
        # recall = recall_score(gold, pred, average='weighted')
        recall = recall_score(gold, pred, average='weighted', labels=labels)

        # f1 = f1_score(gold, pred, average='weighted')
        f1 = f1_score(gold, pred, average='weighted', labels=labels)

        return loss, accuracy, recall, f1
    else:
        return loss


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    if args.problem == 'classification':
        # [max_len * batch_size]
        gold = gold.contiguous().view(-1)

        if smoothing:
            eps = 0.1
            n_classes = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * \
                eps / (n_classes - 1)

            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.sum()  # average later
        else:
            if args.classes_weight is not None and len(args.classes_weight) != 0:
                # weight = torch.tensor(args.classes_weight, device=device)
                loss = F.cross_entropy(pred, gold, weight=args.classes_weight, reduction='sum')
            else:
                loss = F.cross_entropy(pred, gold, reduction='sum')
    else:
        # pred: [batch_size, 1], gold: [batch_size, 1]
        gold = gold.float()
        # print('pred: ', pred)
        # print('gold: ', gold)
        # loss = F.smooth_l1_loss(input=pred, target=gold, reduction='mean')
        loss = F.mse_loss(input=pred, target=gold, reduction='mean')

    return loss


if __name__ == '__main__':
    mode = args.mode
    epochs = args.epochs
    lr = args.lr
    dropout = args.dropout
    text = args.text

    if args.checkpoint:
        print('load checkpoint...')
        checkpoint = torch.load(args.checkpoint)

        model.load_state_dict(checkpoint['model'])
        optimizer._optimizer.load_state_dict(checkpoint['optimizer'])

        args = checkpoint['settings']

        epoch = checkpoint['epoch']
        args.start_epoch = epoch + 1

        valid_loss = checkpoint['valid_loss']
        if args.problem == 'classification':
            valid_accuracy = checkpoint['valid_accuracy']
            valid_recall = checkpoint['valid_recall']
            valid_f1 = checkpoint['valid_f1']

            print('  - (checkpoint) epoch: {epoch: d}, loss: {loss: 8.5f}, '
                'accuracy: {accuracy:3.3f}%, recall: {recall:3.3f}%, f1: {f1:3.3f}%'.format(
                    epoch=epoch,
                    loss=valid_loss,
                    accuracy=100*valid_accuracy,
                    recall=100*valid_recall,
                    f1=100*valid_f1))
        else:
            print('  - (checkpoint) epoch: {epoch: d}, loss: {loss: 8.5f}, '.format(
                    epoch=epoch,
                    loss=valid_loss))

        args.log_mode = 'a'


    args.mode = mode
    args.epochs = epochs
    args.lr = lr
    args.dropout = dropout
    args.text = text

    if args.mode == 'train':
        train_epochs()
    elif args.mode == 'eval':
        eval(epoch)
    elif args.mode == 'test':
        test()
    else:
        raise ValueError('%s is invalid' % mode)
