import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelConfig:
    """ Conditional GAN Model Config """
    latent_size = None
    discriminator_first_hidden_size = None
    discriminator_second_hidden_size = None
    discriminator_dropout = None
    generator_first_hidden_size = None
    generator_second_hidden_size = None
    negative_slope = None
    image_size = None
    n_classes = None
    label_embed_size = None

    def __init__(self, config):
        for k, v in config.items():
            setattr(self, k, v)

        if not self.latent_size:
            logger.error("latent_size is not implemented")
            raise NotImplementedError

        if not self.discriminator_first_hidden_size:
            logger.error("discriminator_first_hidden_size is not implemented")
            raise NotImplementedError

        if not self.discriminator_second_hidden_size:
            logger.error("discriminator_second_hidden_size is not implemented")
            raise NotImplementedError

        if not self.discriminator_dropout:
            logger.error("discriminator_dropout is not implemented")
            raise NotImplementedError

        if not self.generator_first_hidden_size:
            logger.error("generator_first_hidden_size is not implemented")
            raise NotImplementedError

        if not self.generator_second_hidden_size:
            logger.error("generator_second_hidden_size is not implemented")
            raise NotImplementedError

        if not self.negative_slope:
            logger.error("negative_slope is not implemented")
            raise NotImplementedError

        if not self.image_size:
            logger.error("image_size is not implemented")
            raise NotImplementedError

        if not self.n_classes:
            logger.error("n_classes is not implemented")
            raise NotImplementedError

        if not self.label_embed_size:
            logger.error("label_embed_size is not implemented")
            raise NotImplementedError


class LinearGenerator(nn.Module):
    """
    Generator with Linear Layers and Condition
    input : Gaussian Random Noise z
    output : Generated Image
    """

    def __init__(self, config):
        super(LinearGenerator, self).__init__()

        self.image_len = int(config.image_size ** 0.5)

        self.label_embed = nn.Embedding(config.n_classes, config.label_embed_size)

        self.generator = nn.Sequential(
            nn.Linear(config.latent_size + config.label_embed_size, config.generator_first_hidden_size),
            nn.LeakyReLU(config.negative_slope),
            nn.Linear(config.generator_first_hidden_size, config.generator_second_hidden_size),
            nn.LeakyReLU(config.negative_slope),
            nn.Linear(config.generator_second_hidden_size, config.generator_third_hidden_size),
            nn.LeakyReLU(config.negative_slope),
            nn.Linear(config.generator_third_hidden_size, config.image_size),
            nn.Tanh()
        )

        self.apply(self.init_weights)

        logger.info(f"number of total parameters for G: {sum(p.numel() for p in self.parameters())}")

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            nn.init.xavier_normal(module.weight)

    def forward(self, z, y):
        x = torch.cat([self.label_embed(y), z], dim=-1)
        return self.generator(x).view(-1, 1, self.image_len, self.image_len)


class LinearDiscriminator(nn.Module):
    """
    Discriminator with Linear Layers and Condition
    input : Image
    output : 0~1 float (0: Fake Image, 1: Real Image)
    """

    def __init__(self, config):
        super(LinearDiscriminator, self).__init__()

        self.image_size = config.image_size

        self.label_embed = nn.Embedding(config.n_classes, config.label_embed_size)

        self.discriminator = nn.Sequential(
            nn.Linear(config.image_size + config.label_embed_size, config.discriminator_first_hidden_size),
            nn.LeakyReLU(config.negative_slope),
            nn.Dropout(config.discriminator_dropout),
            nn.Linear(config.discriminator_first_hidden_size, config.discriminator_second_hidden_size),
            nn.LeakyReLU(config.negative_slope),
            nn.Dropout(config.discriminator_dropout),
            nn.Linear(config.discriminator_second_hidden_size, config.discriminator_third_hidden_size),
            nn.LeakyReLU(config.negative_slope),
            nn.Dropout(config.discriminator_dropout),
            nn.Linear(config.discriminator_third_hidden_size, 1),
            nn.Sigmoid()
        )

        self.apply(self.init_weights)

        logger.info(f"number of parameters for D: {sum(p.numel() for p in self.parameters())}")

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            nn.init.xavier_normal(module.weight)

    def forward(self, x, y):
        x = torch.cat([self.label_embed(y), x.view(-1, self.image_size), ], dim=-1)
        return self.discriminator(x)


class ConvolutionGenerator(nn.Module):

    def __init__(self, config):
        super(ConvolutionGenerator, self).__init__()

        self.image_len = int(config.image_size ** 0.5)

        self.label_embed = nn.Embedding(config.n_classes, config.label_embed_size)
        self.generator_linear_hidden_size = config.generator_linear_hidden_size

        self.linear = nn.Sequential(
            nn.Linear(config.latent_size + config.label_embed_size, self.generator_linear_hidden_size),
            nn.LeakyReLU(config.negative_slope),
            nn.Linear(self.generator_linear_hidden_size, self.generator_linear_hidden_size),
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=config.generator_linear_hidden_size,
                               out_channels=config.generator_conv1[1],
                               kernel_size=config.generator_conv1[2],
                               stride=config.generator_conv1[3],
                               padding=config.generator_conv1[4]),
            # nn.BatchNorm2d(config.generator_conv1[1]),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=config.generator_conv2[0],
                               out_channels=config.generator_conv2[1],
                               kernel_size=config.generator_conv2[2],
                               stride=config.generator_conv2[3],
                               padding=config.generator_conv2[4]),
            # nn.BatchNorm2d(config.generator_conv2[1]),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=config.generator_conv3[0],
                               out_channels=config.generator_conv3[1],
                               kernel_size=config.generator_conv3[2],
                               stride=config.generator_conv3[3],
                               padding=config.generator_conv3[4]),
            # nn.BatchNorm2d(config.generator_conv3[1]),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=config.generator_conv4[0],
                               out_channels=config.generator_conv4[1],
                               kernel_size=config.generator_conv4[2],
                               stride=config.generator_conv4[3],
                               padding=config.generator_conv4[4]),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=config.generator_conv5[0],
                               out_channels=config.generator_conv5[1],
                               kernel_size=config.generator_conv5[2],
                               stride=config.generator_conv5[3],
                               padding=config.generator_conv5[4]),            
            nn.Tanh()
        )

        self.apply(self.init_weights)

        logger.info(f"number of total parameters for G: {sum(p.numel() for p in self.parameters())}")

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            nn.init.xavier_normal(module.weight)

        elif isinstance(module, nn.ConvTranspose2d):
            nn.init.xavier_normal(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.BatchNorm1d):
            nn.init.xavier_normal(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.BatchNorm2d):
            nn.init.xavier_normal(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x, y):
        x = torch.cat([self.label_embed(y), x], dim=-1)
        return self.conv(self.linear(x).view(-1, self.generator_linear_hidden_size, 1, 1))


class ConvolutionDiscriminator(nn.Module):
    """
    Discriminator with Convolution Layers
    input : Image
    output : 0~1 float (0: Fake Image, 1: Real Image)
    """

    def __init__(self, config):
        super(ConvolutionDiscriminator, self).__init__()

        self.image_size = config.image_size
        self.label_embed = nn.Embedding(config.n_classes, config.label_embed_size)

        self.linear = nn.Sequential(
            nn.Linear(512 + config.label_embed_size, config.discriminator_linear1),
            nn.LeakyReLU(config.negative_slope),
            nn.Linear(config.discriminator_linear1, config.discriminator_linear2),
            nn.LeakyReLU(config.negative_slope),
            nn.Linear(config.discriminator_linear2, 1),
            nn.Sigmoid()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=config.discriminator_conv1[0],
                      out_channels=config.discriminator_conv1[1],
                      kernel_size=config.discriminator_conv1[2],
                      stride=config.discriminator_conv1[3],
                      padding=config.discriminator_conv1[4]),
            nn.LeakyReLU(config.negative_slope),
            nn.Conv2d(in_channels=config.discriminator_conv2[0],
                      out_channels=config.discriminator_conv2[1],
                      kernel_size=config.discriminator_conv2[2],
                      stride=config.discriminator_conv2[3],
                      padding=config.discriminator_conv2[4]),
            # nn.BatchNorm2d(config.discriminator_conv2[1]),
            nn.LeakyReLU(config.negative_slope),
            nn.Conv2d(in_channels=config.discriminator_conv3[0],
                      out_channels=config.discriminator_conv3[1],
                      kernel_size=config.discriminator_conv3[2],
                      stride=config.discriminator_conv3[3],
                      padding=config.discriminator_conv3[4]),
            # nn.BatchNorm2d(config.discriminator_conv3[1]),
            nn.LeakyReLU(config.negative_slope),
            nn.Conv2d(in_channels=config.discriminator_conv4[0],
                      out_channels=config.discriminator_conv4[1],
                      kernel_size=config.discriminator_conv4[2],
                      stride=config.discriminator_conv4[3],
                      padding=config.discriminator_conv4[4]),
            # nn.BatchNorm2d(config.discriminator_conv4[1]),
            nn.LeakyReLU(config.negative_slope),
            nn.Conv2d(in_channels=config.discriminator_conv5[0],
                      out_channels=config.discriminator_conv5[1],
                      kernel_size=config.discriminator_conv5[2],
                      stride=config.discriminator_conv5[3],
                      padding=config.discriminator_conv5[4]),
            nn.MaxPool2d(2),
        )

        logger.info(f"number of total parameters for D: {sum(p.numel() for p in self.parameters())}")

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_normal(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
               

        elif isinstance(module, nn.BatchNorm2d):
            nn.init.xavier_normal(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            nn.init.xavier_normal(module.weight)

    def forward(self, x, y):
        batch_size = x.size(0)
        embed_label = self.label_embed(y)
        embed_image = self.conv(x).view(batch_size, -1)  # flatten

        x = torch.cat([embed_image, embed_label], dim=-1)
        out = self.linear(x)
        return out
