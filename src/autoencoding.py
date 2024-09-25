import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights

# import perceptual_similarity.dist_model as dm


class MNISTAutoencoder(nn.Module):
    """
    Source: https://github.com/casey-meehan/data-copying
    Paper: http://proceedings.mlr.press/v108/meehan20a/meehan20a-supp.pdf
    """

    def __init__(self, lam=0.1, d=4):
        super(MNISTAutoencoder, self).__init__()
        self.d = d
        self.lam = lam
        self.encoder = nn.Sequential(
            nn.Conv2d(1, d * 2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(d * 2),
            nn.Conv2d(d * 2, d * 2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(d * 2),
            nn.Conv2d(d * 2, d * 4, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(d * 4),
            nn.Conv2d(d * 4, d * 4, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        # then flatten features
        self.bottle = nn.Sequential(
            nn.Linear(d * 4 * 7 * 7, d * 4 * 7 * 7),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(d * 4 * 7 * 7, d * 4 * 4),
        )

        self.unbottle = nn.Sequential(
            nn.Linear(d * 4 * 4, d * 4 * 7 * 7),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(d * 4 * 7 * 7, d * 4 * 7 * 7),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.BatchNorm2d(d * 4),
            nn.Conv2d(d * 4, d * 4, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(d * 4),
            nn.Conv2d(d * 4, d * 2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.BatchNorm2d(d * 2),
            nn.Conv2d(d * 2, d * 2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(d * 2),
            nn.Conv2d(d * 2, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

        ### since we just use the weights given by Meehan et al (2020) we don't need this

        # #Build perceptual loss vgg model
        # self.vgg_percep = dm.DistModel()
        # #(load vgg network weights -- pathing is relative to top level)
        # self.vgg_percep.initialize(model='net-lin', net='vgg',
        #     model_path = './MNIST_autoencoder/perceptual_similarity/weights/vgg.pth', use_gpu=True, spatial=False)

    def forward(self, x):
        z = self.encode(x)
        x_ = self.decode(z)
        return x_, z

    def encode(self, x):
        batch_size = x.shape[0]
        enc = self.encoder(x)
        z = self.bottle(enc.view(batch_size, -1))
        return z

    def decode(self, z):
        batch_size = z.shape[0]
        dec = self.unbottle(z)
        x_ = self.decoder(dec.view(batch_size, self.d * 4, 7, 7))
        return x_

    def perceptual_loss(self, im0, im1, z):
        """computes loss as perceptual distance between image x0 and x1
        and adds squared loss of the latent z norm"""
        batch_size = im0.shape[0]
        im0 = im0.expand(batch_size, 3, 28, 28)
        im1 = im1.expand(batch_size, 3, 28, 28)
        percep_dist = self.vgg_percep.forward_pair(im0, im1)
        z_norms = (z.view(batch_size, -1) ** 2).sum(dim=1)
        latent_penalty = self.lam * (F.relu(z_norms - 1))
        return (percep_dist + latent_penalty).sum()


class InceptionV3(nn.Module):
    """
    Source: https://github.com/casey-meehan/data-copying
    Paper: http://proceedings.mlr.press/v108/meehan20a/meehan20a-supp.pdf
    """

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,  # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3,  # Final average pooling features
    }

    def __init__(
        self,
        output_blocks=[DEFAULT_BLOCK_INDEX],
        resize_input=True,
        normalize_input=True,
        requires_grad=False,
    ):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradient. Possibly useful
            for finetuning the network
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, "Last possible output block index is 3"

        self.blocks = nn.ModuleList()

        inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2),
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp
