import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        m = []
        #l1
        m.append(nn.Conv2d(in_size[0], 32, kernel_size=5))
        m.append(nn.BatchNorm2d(32))
        m.append(nn.ReLU())
        #l2
        m.append(nn.Conv2d(32, 128, kernel_size=5, padding=2, stride=2))
        m.append(nn.BatchNorm2d(128))
        m.append(nn.LeakyReLU(0.5))
        #l3
        m.append(nn.Conv2d(128, 256, kernel_size=5, padding=2, stride=2))
        m.append(nn.BatchNorm2d(256))
        m.append(nn.Tanh())
        #l4
        m.append(nn.Conv2d(256, 256, kernel_size=5, padding=2, stride=2))
        m.append(nn.BatchNorm2d(256))
        m.append(nn.LeakyReLU(0.5))
        #l5
        m.append(nn.Conv2d(256, 1, kernel_size=8))
        m.append(nn.LeakyReLU(0.5))

        self.dsc_cnn = nn.Sequential(*m)
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        y = self.dsc_cnn(x).reshape(-1,1)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        self.featuremap_size = featuremap_size

        m = []
        #L1
        m.append(nn.ConvTranspose2d(z_dim, featuremap_size, kernel_size=4, bias=False))
        m.append(nn.BatchNorm2d(featuremap_size))
        m.append(nn.LeakyReLU(0.5))
        #L2
        m.append(nn.ConvTranspose2d(featuremap_size, 256, kernel_size=5, padding=2, stride=4, bias=False))
        m.append(nn.BatchNorm2d(256))
        m.append(nn.Dropout2d(0.1))
        m.append(nn.LeakyReLU(0.5))
        #L3
        m.append(nn.ConvTranspose2d(256, 512, kernel_size=5, padding=1, stride=5, output_padding=1, bias=False))
        m.append(nn.BatchNorm2d(512))
        m.append(nn.Dropout2d(0.3))
        m.append(nn.Tanh())
        #L4
        m.append(nn.Conv2d(512, 128, kernel_size=3, padding=1, bias=False))
        m.append(nn.BatchNorm2d(128))
        m.append(nn.LeakyReLU(0.5))
        #L5
        m.append(nn.Conv2d(128, out_channels, kernel_size=3, padding=1, bias=False))
        m.append(nn.LeakyReLU(0.5))

        self.gen_cnn = nn.Sequential(*m)
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        with torch.set_grad_enabled(with_grad):
            z = torch.randn_like(torch.zeros(size=(n, self.z_dim)), device=device)
            samples = self.forward(z=z)
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        x = torch.tanh(self.gen_cnn(z.reshape(z.shape[0], z.shape[1], 1, 1)))
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss. Apply noise to both the real data and the
    #  generated labels.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    def tensor_gen(y, d_l, l_n, dev):
        return (d_l * torch.ones_like(y, device=dev)) + (l_n * torch.rand_like(y, device=dev)) - (l_n / 2)

    device = y_data.device
    data_tensor = tensor_gen(y_data, data_label, label_noise, device)
    gen_tensor = tensor_gen(y_generated, 1-data_label, label_noise, device)

    loss_calc = torch.nn.BCEWithLogitsLoss()
    loss_data = loss_calc(y_data, data_tensor)
    loss_generated = loss_calc(y_generated, gen_tensor)
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    gen_tensor = torch.ones_like(y_generated, device=y_generated.device) * data_label
    loss_calc = nn.BCEWithLogitsLoss()
    loss = loss_calc(y_generated, gen_tensor)
    # ========================
    return loss


def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: Tensor,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    fake_dsc = gen_model.sample(n=x_data.shape[0])
    gen_optimizer.zero_grad()
    dsc_optimizer.zero_grad()

    dsc_loss = dsc_loss_fn(dsc_model(x_data), dsc_model(fake_dsc))
    dsc_loss.backward()

    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    fake_gen = gen_model.sample(n=x_data.shape[0], with_grad=True)
    gen_optimizer.zero_grad()
    dsc_optimizer.zero_grad()

    gen_loss = gen_loss_fn(dsc_model(fake_gen))
    gen_loss.backward()

    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    if len(gen_losses) > 1 and len(dsc_losses) > 1:
            if dsc_losses[-1] < dsc_losses[-2] and gen_losses[-1] < gen_losses[-2]:
                saved = True
                torch.save(gen_model, checkpoint_file)
                print(f"\n*** Saved checkpoint {checkpoint_file}")
    # ========================

    return saved
