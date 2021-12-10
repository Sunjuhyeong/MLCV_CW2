# -*- coding: utf-8 -*-
"""Q3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SVyn9gnvrg05WO-HLLjD9f95tlWTCCRp
"""

import logging
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from torchvision.utils import save_image
import torchvision.datasets
import matplotlib.pyplot as plt

from torchvision import transforms as tf

from dataset import NewSimpleDataset
from model import ModelConfig, ConvolutionGenerator, LinearGenerator
from trainer import TrainConfig
from utils import set_seed
from classifier.model import CNNClassifier, SmallCNN, Linear
from torch.utils.data import DataLoader
from tqdm import trange
from tqdm import tqdm


def classifier_training(train, test, new, real_ratio, dir):
    transforms = tf.Compose([
        tf.Resize(32),
        tf.ToTensor()
    ])
    batch_size = 8

    mnist_test = torchvision.datasets.MNIST(root='data/',
                                            train=False,
                                            transform=transforms,
                                            download=True)

    test_loader = DataLoader(dataset=test,
                             batch_size=8,
                             shuffle=False,
                             drop_last=False)

    # model = Linear()
    model = CNNClassifier()
    model = model.cuda(0)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    images = new.images
    label = new.labels
    n = images.size(0)
    best_accuracy = 0.
    for epoch in trange(100):

        model.train()
        x, y = None, None
        # pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        # for step, (x, y) in pbar:
        max = (n // batch_size)
        if n % batch_size != 0:
            max += 1
        model.eval()
        for i in range(max):
            if i == max - 1:
                x, y = images[batch_size * i:n, :, :, :].clone().detach(), label[batch_size * i:n].clone().detach()
            else:
                x, y = images[batch_size * i:batch_size * (i + 1), :, :, :].clone().detach(), label[
                                                                                              batch_size * i:batch_size * (
                                                                                                          i + 1)].clone().detach()
            x.cuda(0), y.cuda(0)
            # save_image(x,
            #            os.path.join("figure", "new_image_train_.png"))
            # print("label:", y)
            optimizer.zero_grad()
            model = model.to(device)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        correct = 0
        model.eval()
        for (x, y) in test_loader:
            x, y = x.cuda(0), y.cuda(0)
            # save_image(x,
            #            os.path.join("figure", "new_image_train_.png"))
            # print("label:", y)
            with torch.no_grad():
                logits = model(x)
                correct += (torch.argmax(logits, 1) == y).sum()

        accuracy = correct / len(mnist_test)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'checkpoints_cnn_cat_' + dir + f'/best_{real_ratio}.pt')
            print(f'[Epoch : {epoch}/100] Best Accuracy : {accuracy:.6f}%')


def testing(model, test, real_ratio):
    batch_size = 8
    test_loader = DataLoader(dataset=test,
                             batch_size=batch_size,
                             shuffle=False,
                             drop_last=False)

    model = model.cuda(0)
    fail_images = torch.Tensor()
    fail_images = fail_images.to(device)
    mis_predicted = torch.Tensor()
    mis_predicted = mis_predicted.to(device)

    correct = 0
    model.eval()
    for (x, y) in test_loader:
        x, y = x.cuda(0), y.cuda(0)

        with torch.no_grad():
            logits = model(x)
            predicted = torch.argmax(logits, 1)
            correct += (predicted == y).sum()
            fail_idx = predicted != y
            fail_images = torch.cat((fail_images, torch.masked_select(y, fail_idx)), dim=0)
            mis_predicted = torch.cat((mis_predicted, torch.masked_select(predicted, fail_idx)), dim=0)
    # save_image(torch.masked_select(x, fail_idx),
    #            os.path.join("figure", "fail_images.png"))
    # print("label:", torch.masked_select(y, fail_idx))

    # _, counts = torch.unique(fail_images, sorted=True, return_counts=True)
    # _, predicted_counts = torch.unique(mis_predicted, sorted=True, return_counts=True)
    counts = torch.Tensor(10)
    predicted_counts = torch.Tensor(10)
    for i in range(10):
        counts[i] = (fail_images == i).sum()
        predicted_counts[i] = (mis_predicted == i).sum()
    print(f"{real_ratio} Fail_example_counts :\n", counts)
    print(f"{real_ratio} Fail_example_predicted_counts :\n", predicted_counts)

    accuracy = correct / len(mnist_test)
    print(f'Accuracy : {accuracy:.6f}%')
    return counts.to(device), predicted_counts.to(device)

def denormalize(images):
    out = (images + 1) / 2
    return out.clamp(0, 1)


"""# Make Dataset"""


def make_dataset(mode, images, labels, conf, fake_images=None, fake_labels=None, fake_conf=None, real_ratio=1.0):
    # Images: (60000, 28, 28)
    # labels: (60000, 1)
    # conf:   (60000, 1)
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.cuda.current_device()

    N = labels.size(0)
    fake_N = fake_labels.size(0)

    # """Shuffle the fake dataset"""
    indices = torch.randperm(fake_N)
    fake_labels = fake_labels[indices]
    fake_conf = fake_conf[indices]
    fake_images = fake_images[indices]

    """Combine the 2 Dataset"""
    idx = int(real_ratio * N)
    fake_idx = int((1 - real_ratio) * N)
    image_data = images[:idx, :, :].to(device)
    image_data = torch.unsqueeze(image_data, dim=1)
    label_data = labels[:idx].to(device)
    conf = conf[:idx].to(device)
    _, counts = torch.unique(label_data, sorted=True, return_counts=True)
    print("counts", counts)
    save_image(denormalize(image_data[0:5, :, :, :]),
               os.path.join("figure", "real_image_train_.png"))

    fake_imageData = fake_images[:fake_idx, :, :].to(device)
    fake_imageData = torch.unsqueeze(fake_imageData, dim=1)
    fake_labelData = fake_labels[:fake_idx].to(device)
    fake_conf = fake_conf[:fake_idx].to(device)
    _, fake_counts = torch.unique(fake_labelData, sorted=True, return_counts=True)
    print("counts", fake_counts)

    new_images = torch.cat((image_data, fake_imageData), dim=0)
    new_labels = torch.cat((label_data, fake_labelData), dim=0)
    new_confs = torch.cat((conf, fake_conf), dim=0)
    new_labels = new_labels.to(torch.int64)

    if mode == 0:
        """Mixing"""
        indices = torch.randperm(N)
        new_images = new_images[indices]
        new_labels = new_labels[indices]
    elif mode == 1:
        """Inverse Sort by confidence"""
        # n = torch.argsort(-new_confs).to(device) # Descending order
        n = torch.argsort(new_confs).to(device)    # Ascending order
        new_images = new_images[n, :, :, :]
        new_labels = new_labels[n]
    elif mode == 2:
        """Sort by confidence"""
        n = torch.argsort(-new_confs).to(device) # Descending order
        # n = torch.argsort(new_confs).to(device)    # Ascending order
        new_images = new_images[n, :, :, :]
        new_labels = new_labels[n]
        # _, front_counts = torch.unique(new_labels[:10000], sorted=True, return_counts=True)
        # _, back_counts = torch.unique(new_labels[50000:], sorted=True, return_counts=True)
        #
        # x1 = np.linspace(0, 9, 10)
        # x1 = x1.astype(int)
        # plt.subplot(1, 2, 1)  # nrows=2, ncols=1, index=1
        # # front_counts.cpu()
        # plt.bar(x1, front_counts.cpu())
        # plt.title('Front 10000 labels')
        # plt.ylabel('Number of images')
        # plt.xticks(x1, x1)
        #
        # # back_counts.cpu()
        # plt.subplot(1, 2, 2)  # nrows=2, ncols=1, index=1
        # plt.bar(x1, back_counts.cpu())
        # plt.title('Back 10000 labels')
        # plt.xlabel('Number')
        # plt.ylabel('Number of images')
        # plt.xticks(x1, x1)
        #
        # plt.tight_layout()
        # plt.savefig('figure/confidence_distribution.png')

    dataset = NewSimpleDataset(new_images, new_labels)
    save_image(new_images[N // 2 - 20:N // 2, :, :, :],
               os.path.join("figure", "new_image_train_.png"))
    print(new_labels[N // 2 - 20:N // 2])
    return dataset




def cat_dataset(mode, images, labels, conf, fake_images=None, fake_labels=None, fake_conf=None, fake_ratio=1.0):
    # Images: (60000, 28, 28)
    # labels: (60000, 1)
    # conf:   (60000, 1)
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.cuda.current_device()

    N = labels.size(0)
    fake_N = fake_labels.size(0)

    # """Shuffle the fake dataset"""
    indices = torch.randperm(fake_N)
    fake_labels = fake_labels[indices]
    fake_conf = fake_conf[indices]
    fake_images = fake_images[indices]

    """Combine the 2 Dataset"""
    # idx = int(real_ratio * N)
    # fake_idx = int((1 - real_ratio) * N)
    idx = int(N)
    fake_idx = int(fake_ratio * N)
    image_data = images[:idx, :, :].to(device)
    image_data = torch.unsqueeze(image_data, dim=1)
    label_data = labels[:idx].to(device)
    conf = conf[:idx].to(device)
    _, counts = torch.unique(label_data, sorted=True, return_counts=True)
    print("counts", counts)
    save_image(denormalize(image_data[0:5, :, :, :]),
               os.path.join("figure", "real_image_train_.png"))

    fake_imageData = fake_images[:fake_idx, :, :].to(device)
    fake_imageData = torch.unsqueeze(fake_imageData, dim=1)
    fake_labelData = fake_labels[:fake_idx].to(device)
    fake_conf = fake_conf[:fake_idx].to(device)
    _, fake_counts = torch.unique(fake_labelData, sorted=True, return_counts=True)
    print("counts", fake_counts)

    new_images = torch.cat((image_data, fake_imageData), dim=0)
    new_labels = torch.cat((label_data, fake_labelData), dim=0)
    new_confs = torch.cat((conf, fake_conf), dim=0)
    new_labels = new_labels.to(torch.int64)

    """Sort by confidence"""
    # n = torch.argsort(-new_confs).to(device)
    # new_images = new_images[n, :, :, :]
    # new_labels = new_labels[n]
    # new_labels = new_labels.to(torch.int64)

    if mode == 0:
        """Mixing"""
        indices = torch.randperm(N)
        new_images = new_images[indices]
        new_labels = new_labels[indices]
    elif mode == 1:
        """Inverse Sort by confidence"""
        # n = torch.argsort(-new_confs).to(device) # Descending order
        n = torch.argsort(new_confs).to(device)    # Ascending order
        new_images = new_images[n, :, :, :]
        new_labels = new_labels[n]
    elif mode == 2:
        """Sort by confidence"""
        n = torch.argsort(-new_confs).to(device) # Descending order
        # n = torch.argsort(new_confs).to(device)    # Ascending order
        new_images = new_images[n, :, :, :]
        new_labels = new_labels[n]


    dataset = NewSimpleDataset(new_images, new_labels)
    save_image(new_images[N // 2 - 20:N // 2, :, :, :],
               os.path.join("figure", "new_image_train_.png"))
    print(new_labels[N // 2 - 20:N // 2])
    return dataset


"""# Generate Images """


def generate_images(generator, config, n=60000):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        # generator = torch.nn.DataParallel(generator.to(device))
    generator = generator.to(device)

    ones = torch.ones(n // 10)
    ones = ones.long()
    ones = ones.to(device)
    fake_images = torch.Tensor()
    for i in range(10):
        z = torch.randn(n // 10, config.latent_size).to(device)
        with torch.no_grad():
            fake_images = fake_images.to(device)
            fake_images = torch.cat([fake_images, generator(z, ones * i)], dim=0)
    fake_images = torch.squeeze(fake_images, dim=1)
    fake_y = torch.cat([ones * 0, ones * 1, ones * 2, ones * 3, ones * 4,
                        ones * 5, ones * 6, ones * 7, ones * 8, ones * 9], dim=0).long().to(device)

    return fake_images, fake_y


"""# Get Confidence"""


def classify(model, input_images):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.cuda.current_device()

    model = model.to(device)
    images = input_images.clone().detach()
    images = images.type(torch.FloatTensor)
    images = images.to(device)
    images = images.unsqueeze(dim=1)
    n = images.size(0)
    batch_size = 8
    num_class = 10
    conf = torch.zeros(n, num_class)
    labels = torch.zeros(n)
    max = (n // batch_size)
    if n % batch_size != 0:
        max += 1

    soft_max = nn.Softmax(dim=1).to(device)

    model.eval()
    with torch.no_grad():
        for i in range(max):
            if i == max - 1:
                temp_conf = soft_max(model(images[batch_size * i:n, :, :, :]))
                conf[batch_size * i:n, :] = temp_conf
                labels[batch_size * i:n] = torch.argmax(temp_conf, 1)
                break
            else:
                temp_conf = soft_max(model(images[batch_size * i:batch_size * (i + 1), :, :, :]))
                conf[batch_size * i:batch_size * (i + 1), :] = temp_conf
                labels[batch_size * i:batch_size * (i + 1)] = torch.argmax(temp_conf, 1)

    return conf.to(device), labels.to(device)


"""# Main"""

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    set_seed(42)
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.cuda.current_device()

    tyaml = yaml.load(open(f'config/train.yaml', 'r'), Loader=yaml.FullLoader)
    tconf = TrainConfig(tyaml)

    images = np.load(f'data/{tyaml["dataset"]}_images.npy')
    labels = np.load(f'data/{tyaml["dataset"]}_labels.npy')

    transf = tf.Compose([
        tf.Resize(32),
        tf.ToTensor()
    ])
    mnist_train = torchvision.datasets.MNIST(root='data/',
                                             train=True,
                                             transform=transf,
                                             download=True)

    mnist_test = torchvision.datasets.MNIST(root='data/',
                                            train=False,
                                            transform=transf,
                                            download=True)
    train_loader = DataLoader(dataset=mnist_train,
                              batch_size=8,
                              shuffle=False,
                              drop_last=True)

    test_loader = DataLoader(dataset=mnist_test,
                             batch_size=8,
                             shuffle=False,
                             drop_last=False)

    # """Evaluation"""

    # ratio = [0.5]
    ratio = [1.0]
    counts = torch.Tensor(10)
    counts = counts.to(device)
    predicted_counts = torch.Tensor(10)
    predicted_counts = predicted_counts.to(device)
    _, num_labels = torch.unique(mnist_test.targets, sorted=True, return_counts=True)
    num_labels = num_labels.to(device)
    for r in ratio:
        model = CNNClassifier()
        model.load_state_dict(torch.load(
            f'checkpoints_cnn_mixed/best_{r}.pt'))
        temp_counts, temp_predicted = testing(model, mnist_test, r)
        counts += torch.div(temp_counts, num_labels)
        predicted_counts += torch.div(temp_predicted, num_labels)

    x1 = np.linspace(0, 9, 10)
    x1 = x1.astype(int)
    plt.subplot(1, 2, 1)  # nrows=2, ncols=1, index=1
    plt.bar(x1, counts.cpu())
    plt.title('Missed labels')
    plt.ylabel('Ratio')
    plt.xticks(x1, x1)

    plt.subplot(1, 2, 2)  # nrows=2, ncols=1, index=1
    plt.bar(x1, predicted_counts.cpu())
    plt.title('Predicted labels')
    plt.xlabel('Number')
    plt.ylabel('Ratio')
    plt.xticks(x1, x1)

    plt.tight_layout()
    plt.savefig('figure/cnn_fail_label_count.png')
    print("Total Fail_example_counts :\n", counts)
    print("Total Fail_example_predicted_counts :\n", predicted_counts)

    """Generate images"""
    myaml = yaml.load(open('config/model.yaml', 'r'), Loader=yaml.FullLoader)
    mconf = ModelConfig(myaml)
    linear_generator = LinearGenerator(mconf)
    conv_generator = ConvolutionGenerator(mconf)

    generator = conv_generator
    generator.load_state_dict(torch.load(
        'weights/200/G_200.ckpt'))
    # generator.load_state_dict(torch.load(
    #     'weights/200/G_200.ckpt', map_location=torch.device('cpu')))
    num_fake_images = 60000
    fake_images, fake_y = generate_images(generator, tconf, num_fake_images)
    fake_images = denormalize(fake_images)

    """ Classifier """
    # classifier = Linear()
    # classifier.load_state_dict(torch.load(
    #     'classifier/checkpoints/linear_best.pt'))

    classifier = CNNClassifier()
    classifier.load_state_dict(torch.load(
        'classifier/checkpoints/best.pt'))

    images_for_dataset = tf.Resize(32)(mnist_train.data)
    images_for_dataset = torch.div(images_for_dataset, 255)

    """Get Confidence value"""
    fake_conf, fake_labels = classify(classifier, fake_images)
    wrong_indices = (fake_labels != fake_y).nonzero()
    wrong_conf = torch.max(fake_conf[wrong_indices], dim=1)
    right_indices = (fake_labels == fake_y).nonzero()
    right_indices = torch.squeeze(right_indices, dim=1)
    # save_image(fake_images[wrong_indices, :, :],
    #            os.path.join("figure", "fake_wrong_images.png"))
    # print("wrong label:", fake_labels[wrong_indices])
    # print("wrong y:", fake_y[wrong_indices])
    # print("wrong conf:", torch.max( fake_conf[wrong_indices, :], dim=2).values)

    fake_conf = fake_conf[right_indices, :]
    fake_images = fake_images[right_indices, :, :]
    fake_labels = fake_labels[right_indices]
    # print(f"{wrong_indices.size(0)} wrong labels: ", wrong_indices)

    conf, predicted_labels = classify(classifier, images_for_dataset)
    fake_conf = fake_conf.to(torch.float)
    conf = conf.to(torch.float)
    """Mix the Dataset"""
    # hard_indices = (1.0 > hello).nonzero()
    # hard_index = (0.0 < hello[hard_indices]).nonzero()
    # zero_indices = (hello == 0).nonzero()
    # print("hard examples labels: ", hard_index)
    # print(hello[hard_index])
    # print(hello[zero_indices])
    # print(torch.max(hello[hard_index]), torch.min(hello[hard_index]), torch.mean(hello[hard_index], dim=0))
    # real_ratio = 1.0

    """ Save Conf"""
    # fake_max_conf = torch.max(fake_conf, dim=1).values
    # max_conf = torch.max(conf, dim=1).values
    # conf_val = torch.Tensor(10)
    # fake_conf_val = torch.Tensor(10)
    #
    # for i in range(10):
    #     fake_temp_idx = fake_labels == i
    #     new_label = torch.Tensor(labels)
    #     temp_idx = new_label == i
    #     temp_idx = temp_idx.to(device)
    #     fake_temp_idx = fake_temp_idx.to(device)
    #     temp_conf_i = torch.masked_select(max_conf, temp_idx)
    #     fake_temp_conf_i = torch.masked_select(fake_max_conf, fake_temp_idx)
    #     conf_val[i] = torch.mean(temp_conf_i)
    #     fake_conf_val[i] = torch.mean(fake_temp_conf_i)
    #
    # x1 = np.linspace(0, 9, 10)
    # x1 = x1.astype(int)
    # plt.subplot(1, 2, 1)  # nrows=2, ncols=1, index=1
    # plt.bar(x1, conf_val)
    # plt.title('Real images')
    # plt.ylabel('Confidence score')
    # plt.xticks(x1, x1)
    #
    # plt.subplot(1, 2, 2)  # nrows=2, ncols=1, index=1
    # plt.bar(x1, fake_conf_val)
    # plt.title('Fake images')
    # plt.xlabel('Number')
    # plt.ylabel('Confidence score')
    # plt.xticks(x1, x1)
    #
    # plt.tight_layout()
    # plt.savefig('figure/confidence_per_number.png')

    print("confidence score: ", torch.mean(torch.max(conf, dim=1).values))
    print("fake confidence score: ", torch.mean(torch.max(fake_conf, dim=1).values))
    # real_ratio = 0.1
    # dataset = make_dataset(2, images_for_dataset, mnist_train.targets, torch.max(conf, dim=1).values, fake_images,
    #                        fake_labels, torch.max(fake_conf, dim=1).values, real_ratio)
    #
    # newimages = dataset.images
    # aa = torch.max(mnist_test.data[0])
    # bb = torch.max(newimages[0])
    # new_train_loader = DataLoader(dataset=dataset,
    #                               batch_size=8,
    #                               shuffle=False,
    #                               drop_last=True
    #                               )
    # """Training Classifier """
    # print(f"Start training {real_ratio}")
    # classifier_training(mnist_train, mnist_test, dataset, real_ratio, 'sorted')

    # =====================================================================================
    # ratio = [0.1, 0.5, 1.0]
    mode_name = ['mixed', 'inverse', 'sorted', 'not_sorted']
    for mode in range(4):
        if mode == 0:
            continue
        elif mode == 1:
            continue
        dataset = cat_dataset(mode, images_for_dataset, mnist_train.targets, torch.max(conf, dim=1).values, fake_images,
                               fake_labels, torch.max(fake_conf, dim=1).values, 0.5)
        # fake_ratio = 0.5
        # dataset = cat_dataset(images_for_dataset, mnist_train.targets, torch.max(conf, dim=1).values, fake_images,
        #                        fake_labels, torch.max(fake_conf, dim=1).values, fake_ratio)

        newimages = dataset.images
        aa = torch.max(mnist_test.data[0])
        bb = torch.max(newimages[0])
        new_train_loader = DataLoader(dataset=dataset,
                                      batch_size=8,
                                      shuffle=False,
                                      drop_last=True
                                      )
        """Training Classifier """
        print(f"Start training {0.5} - {mode_name[mode]}")
        classifier_training(mnist_train, mnist_test, dataset, 0.5, mode_name[mode])
