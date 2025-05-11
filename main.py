import os
import math
import random
import argparse
from time import time
import glob
import sys
from pathlib import Path
from typing import Iterable, Optional
import numpy as np
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import timm
from timm.utils import accuracy
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from easydict import EasyDict

start_time = 0
end_time = 0
args = EasyDict({
    "batch_size": 64,
    "epochs": 1,
    "accum_iter": 1,
    "input_size": 224,
    "weight_decay": 0.0001,
    "lr": 0.001,
    "root_path": r"",
    "start_epoch": 0,
    "num_workers": 0,
    "pin_mem": 'store_true',
    "no_pin_mem": 'store_false',
    "device":torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "num_classes": 200,
    "drop_rate": 0.3,
})

def build_transform(is_train, args):
    if is_train:
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5375, 0.4818, 0.4186], std=[0.3169, 0.3124, 0.3151])
            ]
        )
    else:
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    path = os.path.join(args.root_path, 'train' if is_train else 'test')
    dataset = torchvision.datasets.ImageFolder(path, transform=transform)
    return dataset

def build_dataloader():
    dataset_train = build_dataset(is_train=True, args=args)
    dataset_test = build_dataset(is_train=False, args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    return data_loader_train, data_loader_test

def train_one_epoch(args, model, train_loader, test_loader, criterion, optimizer, now_loss):

    model.train()
    model = model.to(args.device)
    loss_sum_train = 0.
    labels = []
    pre_labels = []
    for image, label in train_loader:
        image, label = image.to(args.device), label.to(args.device)
        output = model(image)
        loss = criterion(output, label)
        print("loss:", loss)
        loss_sum_train += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        labels += label.tolist()
        prediction = torch.argmax(output, 1)
        pre_labels += prediction.tolist()

    train_loss = loss_sum_train / len(train_loader)
    train_acc = accuracy_score(pre_labels, labels)
    print(f'train_Loss:{train_loss:.3f},  train_Acc:{train_acc:.3f}')

    with torch.no_grad():
        model.eval()
        model = model.to(args.device)
        loss_sum_test = 0.
        labels = []
        pre_labels = []
        for image, label in test_loader:
            image, label = image.to(args.device), label.to(args.device)
            output = model(image)
            loss = criterion(output, label)
            print("loss:",loss)
            loss_sum_test += loss.item()
            labels += label.tolist()
            prediction = torch.argmax(output, 1)
            pre_labels += prediction.tolist()
        test_loss = loss_sum_test / len(test_loader)
        test_acc = accuracy_score(pre_labels, labels)
        print(f'test_Loss:{test_loss:.3f},  test_Acc:{test_acc:.3f}')

        if now_loss > loss_sum_test:
            now_loss = loss_sum_test
            print("保存模型...")
            torch.save(model, 'best_model.pth')

    val_loss = 0
    val_acc = 0
    return train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, now_loss

def train(args, model, data_loader_train, data_loader_test):
    start_time = time.time()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loss_iter = []
    train_acc_iter = []
    val_loss_iter = []
    val_acc_iter = []
    test_loss_iter = []
    test_acc_iter = []
    now_loss = 1e9

    predict(model, data_loader_test, True)

    for epoch in range(args.start_epoch, args.epochs):

        print(f"Epoch {epoch}")

        lr = args.lr / math.pow(10, int(epoch/20))

        train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, now_loss = train_one_epoch(
            args, model, data_loader_train, data_loader_test, criterion, optimizer, now_loss)

        train_loss_iter.append(train_loss)
        train_acc_iter.append(train_acc)
        val_loss_iter.append(val_loss)
        val_acc_iter.append(val_acc)
        test_loss_iter.append(test_loss)
        test_acc_iter.append(test_acc)

    end_time = time.time()

    plt.title(f"model:{args.model_name}, batch_size:{args.batch_size}, running time:{((end_time-start_time)/60):.1f}min")
    plt.plot(range(epoch+1), train_loss_iter, color='purple', label='train_loss')
    plt.plot(range(epoch+1), train_acc_iter, color='red', label='train_acc')
    plt.plot(range(epoch+1), test_loss_iter, color='blue', label='test_loss')
    plt.plot(range(epoch + 1), test_acc_iter, color='green', label='test_acc')
    plt.legend()

    from datetime import datetime
    time_str = datetime.strftime(datetime.now(), '%m-%d_%H-%M')
    log_dir = os.path.join("..\\results", time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    plt.savefig(os.path.join(log_dir, 'acc_loss.png'))
    plt.show()
    return model

def predict(model, dataloader, is_start):
    start = 'start' if is_start==True else "best"
    model.eval()
    model = model.to(args.device)
    predictions = []
    labels = []
    for image, label in dataloader:
        image = image.to(args.device)
        label = label.to(args.device)
        labels += label.tolist()
        output = model(image)
        prediction = torch.argmax(output, 1)
        predictions += prediction.tolist()

    print(f'{start}---test_dataloader_acc正确率: ', accuracy_score(predictions, labels))
    print(f'{start}---test_dataloader_acc样本数: ', accuracy_score(predictions, labels, normalize=False),
          ', 总数: ', len(labels))

def build_model():
    args.model_name = 'vit_tiny_patch16_224'
    model = timm.create_model(args.model_name)
    """
    model = timm.create_model(args.model_name,
                          pretrained=True,
                          num_classes=args.num_classes,
                          drop_rate=args.drop_rate,
                          drop_path_rate=0.1,
                          pretrained_cfg_overlay=dict(file=''))
    """
    return model

if __name__ == '__main__':
    model = build_model()
    data_loader_train, data_loader_test = build_dataloader()
    train(args, model, data_loader_train, data_loader_test)
    model = torch.load('best_model.pth')
    predict(model, data_loader_test, False)