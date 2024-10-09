import torch
from torch import nn
import numpy as np
from abc import ABC, abstractmethod
from sklearn.datasets import make_moons, make_swiss_roll
from sklearn.neighbors import KernelDensity
from torchvision import datasets, transforms
import sys
import os
from src.autoencoding import MNISTAutoencoder


class GenerativeModel(ABC):
    """
    This class defines the interface for generative models.
    """

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def sample(self, n_samples, **kwargs):
        pass

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)

    def get_params(self):
        return self.__dict__


class Pipeline(GenerativeModel):
    """
    A pipeline is able to perform a sequence of transformations on the data after generating samples.
    Useful if we want to apply random projections or other transformations to the generated samples.
    """

    def __init__(self, generative_model, transformations):
        """
        Initializes the pipeline.

        Parameters
        ----------
        generative_model : GenerativeModel
            The generative model to use.
        transformations : list of callable
            A list of transformations to apply to the generated samples.
        """
        self.q = generative_model
        self.tfs = transformations

    def fit(self, X):
        self.q.fit(X)
        for t in self.tfs:
            t.fit(X)

    def sample(self, n_samples):
        samples = self.q.sample(n_samples)
        for t in self.tfs:
            samples = t.transform(samples)
        return samples


class Mixture(GenerativeModel):

    def __init__(self, rho, q1, q2, params_q1={}, params_q2={}):
        """
        Initializes the mixture model.

        Parameters
        ----------
        q1 : GenerativeModel
            The first generative model to use.
        q2 : GenerativeModel
            The second generative model to use.
        rho : float
            The probability of selecting the first generative model.
        """
        if isinstance(q1, str):
            self.q1 = getattr(sys.modules[__name__], q1)(**params_q1)
        else:
            self.q1 = q1
        if isinstance(q2, str):
            self.q2 = getattr(sys.modules[__name__], q2)(**params_q2)
        else:
            self.q2 = q2
        self.rho = rho

    def __str__(self) -> str:
        return f"Mixture(rho={self.rho}, q1={self.q1}, q2={self.q2})"

    def fit(self, X=None):
        """
        Fits the mixture model.

        Parameters
        ----------
        X : np.ndarray
            The data to fit the model to.

        Returns
        -------
        self
            The fitted model.
        """
        self.q1.fit(X)
        self.q2.fit(X)
        return self

    def sample(self, n_samples, **kwargs):
        """
        Samples from the mixture model.

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.

        Returns
        -------
        np.ndarray
            An array of shape (n_samples, n_features) containing the generated samples.
        """
        n_samples_q1 = int(n_samples * self.rho)
        n_samples_q2 = n_samples - n_samples_q1
        samples_q1 = self.q1.sample(n_samples_q1, **kwargs)
        samples_q2 = self.q2.sample(n_samples_q2, **kwargs)
        return np.concatenate([samples_q1, samples_q2])


class Halfmoons(GenerativeModel):

    def __init__(self, noise=0.1):
        """
        Initializes the halfmoons dataset

        Parameters
        ----------
        noise : float, optional
            Standard deviation of Gaussian noise added to the data, by default 0.1
        """
        self.noise = noise

    def __str__(self) -> str:
        return f"Halfmoons(noise={self.noise})"

    def fit(self, X=None):
        return self

    def get_train_and_test_data(self, N_train=2000, N_test=1000):
        """
        Generates training and test data from the halfmoons dataset

        Parameters
        ----------
        N_train : int, optional
            Number of training samples to generate, by default 2000

        N_test : int, optional
            Number of test samples to generate, by default 1000

        Returns
        -------
        tuple (X_train, y_train, X_test, y_test)
            Tuple containing the training and test data
        """
        X_train, y_train = make_moons(n_samples=N_train, noise=self.noise)
        X_test, y_test = make_moons(n_samples=N_test, noise=self.noise)
        return X_train, y_train, X_test, y_test

    def sample(self, n_samples=1000, **kwargs):
        """
        Generates a sample of data points from the halfmoons dataset

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate, by default 1000

        Returns
        -------
        np.ndarray
            Array containing the generated samples
        """
        X, _ = make_moons(n_samples=n_samples, noise=self.noise)
        return X


class Sphere(GenerativeModel):
    """
    Generates samples of the surface of a d-dimensional sphere
    """

    def __init__(self, d):
        self.d = d  # dimension of the sphere

    def __str__(self) -> str:
        return f"Sphere(d={self.d})"

    def fit(self, X=None):
        pass

    def sample(self, n_samples, **kwargs):
        """
        Generates a sample of data points from the surface of a d-dimensional sphere

        Parameters
        ----------
        n_samples : int
            Number of samples to generate

        Returns
        -------
        np.ndarray
            Array containing the generated samples
        """
        samples = np.random.normal(size=(n_samples, self.d))
        samples /= np.linalg.norm(samples, axis=1)[:, None]
        return samples


class MNIST(GenerativeModel):

    def __init__(self, root):
        """
        Initializes the MNIST dataset

        Parameters
        ----------
        root : str
            Path to the data directory
        """
        self.root = root
        self.fitted = False

    def __str__(self) -> str:
        return "MNIST"

    def __repr__(self) -> str:
        return "MNIST"

    def fit(self, X=None):
        if not self.fitted:
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            )
            train_data = datasets.MNIST(
                root=self.root, train=True, download=True, transform=transform
            )
            self.test_data = datasets.MNIST(
                root=self.root, train=False, download=True, transform=transform
            )
            # split train data into train and validation
            self.train_data, self.val_data = torch.utils.data.random_split(
                train_data, [50000, 10000]
            )
            self.fitted = True
        return self

    def get_train_and_test_data(self):
        """
        Loads the MNIST dataset from torchvision.datasets.MNIST

        Returns
        -------
        tuple (X_train, y_train, X_test, y_test)
            Tuple containing the training and test data
        """

        train_loader = torch.utils.data.DataLoader(
            self.train_data, batch_size=50000, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            self.test_data, batch_size=10000, shuffle=False
        )
        X_train, y_train = next(iter(train_loader))
        X_test, y_test = next(iter(test_loader))

        return X_train, y_train, X_test, y_test

    def sample(self, n_samples=50000, S="train", encoded=True):
        """
        Generates a sample of data points from the MNIST dataset

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate, by default 50000

        Returns
        -------
        np.ndarray
            Array containing the generated samples
        """
        if not hasattr(self, "train_data"):
            raise ValueError("Model is not fitted yet. Please call fit() first.")
        if S == "train" and n_samples > 50000:
            raise ValueError("n_samples must be less than or equal to 50000")
        elif (S == "test" or S == "val") and n_samples > 10000:
            raise ValueError("n_samples must be less than or equal to 10000")

        if S == "train":
            data = self.train_data
        elif S == "val":
            data = self.val_data
        elif S == "test":
            data = self.test_data
        else:
            raise ValueError("Invalid value for S. Must be 'train', 'val', or 'test'")

        loader = torch.utils.data.DataLoader(data, batch_size=n_samples, shuffle=True)
        X, _ = next(iter(loader))
        if encoded:
            ae = MNISTAutoencoder(d=4).eval()
            ae.load_state_dict(
                torch.load(
                    f"{self.root}/trained_weights/trained_autoencoder_weights.pth",
                    map_location=torch.device("cpu"),
                    weights_only=True,
                )
            )
            X = ae.encode(X.float()).detach().numpy()

        return X


class FashionMNIST(GenerativeModel):

    def __init__(self, root):
        """
        Initializes the FashionMNIST dataset

        Parameters
        ----------
        root : str
            Path to the data directory
        """
        self.root = root
        self.fitted = False

    def __str__(self) -> str:
        return "FashionMNIST"

    def __repr__(self) -> str:
        return "FashionMNIST"

    def fit(self, X=None):
        if not self.fitted:
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            )
            train_data = datasets.FashionMNIST(
                root=self.root, train=True, download=True, transform=transform
            )
            self.test_data = datasets.FashionMNIST(
                root=self.root, train=False, download=True, transform=transform
            )
            # split train data into train and validation
            self.train_data, self.val_data = torch.utils.data.random_split(
                train_data, [50000, 10000]
            )
            self.fitted = True
        return self

    def get_train_and_test_data(self):
        """
        Loads the FashionMNIST dataset from torchvision.datasets.FashionMNIST

        Returns
        -------
        tuple (X_train, y_train, X_test, y_test)
            Tuple containing the training and test data
        """

        train_loader = torch.utils.data.DataLoader(
            self.train_data, batch_size=50000, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            self.test_data, batch_size=10000, shuffle=False
        )
        X_train, y_train = next(iter(train_loader))
        X_test, y_test = next(iter(test_loader))

        return X_train, y_train, X_test, y_test

    def sample(self, n_samples=50000, S="train"):
        """
        Generates a sample of data points from the FashionMNIST dataset

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate, by default 50000

        Returns
        -------
        np.ndarray
            Array containing the generated samples
        """
        if not hasattr(self, "train_data"):
            raise ValueError("Model is not fitted yet. Please call fit() first.")
        if S == "train" and n_samples > 50000:
            raise ValueError("n_samples must be less than or equal to 50000")
        elif (S == "test" or S == "val") and n_samples > 10000:
            raise ValueError("n_samples must be less than or equal to 10000")

        if S == "train":
            data = self.train_data
        elif S == "val":
            data = self.val_data
        elif S == "test":
            data = self.test_data
        else:
            raise ValueError("Invalid value for S. Must be 'train', 'val', or 'test'")

        loader = torch.utils.data.DataLoader(data, batch_size=n_samples, shuffle=True)
        X, _ = next(iter(loader))
        #### ENCODER MISSING
        # if encoded:
        #     ae = MNISTAutoencoder(d=4).eval()
        #     ae.load_state_dict(
        #         torch.load(
        #             f"{self.root}/trained_weights/trained_autoencoder_weights.pth",
        #             map_location=torch.device("cpu"),
        #             weights_only=True,
        #         )
        #     )
        #     X = ae.encode(X.float()).detach().numpy()

        return X


class KDE(KernelDensity, GenerativeModel):

    def __init__(self, bandwidth=0.1):
        """
        Initializes the kernel density estimator

        Parameters
        ----------
        bandwidth : float, optional
            Bandwidth of the kernel, by default 0.1
        """
        super().__init__(bandwidth=bandwidth)

    def __str__(self) -> str:
        return f"KDE(sigma={self.bandwidth})"

    def fit(self, X):
        super().fit(X)
        return self

    def sample(self, n_samples=1000, **kwargs):
        """
        Generates a sample of data points from the kernel density estimator

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate, by default 1000

        Returns
        -------
        np.ndarray
            Array containing the generated samples
        """
        return super().sample(n_samples)


class MultivariateGaussian(GenerativeModel):

    def __init__(self, dim):
        """
        Multivariate Gaussian distribution with mean 0 and identity covariance matrix.

        Parameters
        ----------
        dim : int
            Dimension of the Gaussian distribution.
        """
        self.dim = dim
        self.mu = np.zeros(dim)
        self.sigma = np.eye(dim)

    def fit(self, X, **kwargs):
        return self

    def sample(self, n):
        return np.random.multivariate_normal(self.mu, self.sigma, n)


class SphereSurfaceCopier:

    def __init__(self, radius=1):
        self.radius = radius

    def fit(self, X):
        self.data = X
        self.d = X.shape[1]
        return self

    def sample(self, n, **kwargs):
        samples = np.random.normal(size=(n, self.d))
        samples = self.radius * samples / np.linalg.norm(samples, axis=1)[:, None]
        idx = np.random.choice(self.data.shape[0], n)
        samples += self.data[idx]
        return samples


class Memorizer(GenerativeModel):

    def __init__(self, radius=0.02, n_copying=20):
        self.r = radius
        self.n_copying = n_copying

    def fit(self, X):
        self.subset = X[np.random.choice(len(X), size=self.n_copying, replace=False)]
        return self

    def sample(self, n, **kwargs):
        if not hasattr(self, "subset"):
            raise ValueError("Model is not fitted yet. Please call fit() first.")
        X = self.subset[np.random.choice(len(self.subset), size=n)]
        dim = X.shape[1]

        rand_dir = np.random.normal(size=(dim, n))
        rand_dir /= np.linalg.norm(rand_dir, axis=0)
        rand_rad = np.random.random(n) ** (1 / dim)
        X += self.r * (rand_rad * rand_dir).T

        return X


class VAE(nn.Module):
    """
    Source: https://github.com/casey-meehan/data-copying
    Paper: http://proceedings.mlr.press/v108/meehan20a/meehan20a-supp.pdf
    """

    def __init__(self, d=50, l=3, root="../data.nosync/", device="cpu"):
        """builds VAE
        Inputs:
            - d: dimension of latent space
            - l: number of layers
        """
        super(VAE, self).__init__()
        self.d = d
        self.root = root
        self.device = device

        # Build VAE here
        self.encoder, self.decoder, self.encoder_mean, self.encoder_lv = self.build_VAE(
            d, l
        )

    def __str__(self) -> str:
        return f"VAE(d={self.d}, l={self.l})"

    def build_VAE(self, d, l):
        """builds VAE with specified latent dimension and number of layers
        Inputs:
            -d: latent dimension
            -l: number of layers
        """
        encoder_layers = []
        decoder_layers = []
        alpha = 3 / l

        for lyr in range(l)[::-1]:
            lyr += 1
            dim_a = int(np.ceil(2 ** (alpha * (lyr + 1))))
            dim_b = int(np.ceil(2 ** (alpha * lyr)))
            if lyr == l:
                encoder_layers.append(nn.Linear(784, d * dim_b))
                encoder_layers.append(nn.ReLU())
                decoder_layers.insert(0, nn.Linear(d * dim_b, 784))
                decoder_layers.insert(0, nn.ReLU())
            else:
                encoder_layers.append(nn.Linear(d * dim_a, d * dim_b))
                encoder_layers.append(nn.ReLU())
                decoder_layers.insert(0, nn.Linear(d * dim_b, d * dim_a))
                decoder_layers.insert(0, nn.ReLU())
        decoder_layers.insert(0, nn.Linear(d, d * int(np.ceil(2 ** (alpha)))))

        encoder = nn.Sequential(*encoder_layers)
        decoder = nn.Sequential(*decoder_layers)

        encoder_mean = nn.Linear(d * int(np.ceil(2 ** (alpha))), d)
        encoder_lv = nn.Linear(d * int(np.ceil(2 ** (alpha))), d)
        return encoder, decoder, encoder_mean, encoder_lv

    def fit(self, X, path=None):
        if path is not None:
            self.load_state_dict(
                torch.load(
                    path, map_location=torch.device(self.device), weights_only=True
                )
            )
        return self

    def encode(self, x):
        """take an image, and return latent space mean + log variance
        Inputs:
            -images, x, flattened to 784
        Outputs:
            -means in latent dimension
            -logvariances in latent dimension
        """
        h1 = self.encoder(x)
        return self.encoder_mean(h1), self.encoder_lv(h1)

    def reparameterize(self, mu, logvar):
        """Sample in latent space according to mean and logvariance
        Inputs:
            -mu: batch of means
            -logvar: batch of logvariances
        Outputs:
            -samples: batch of latent samples
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent space samples
        Inputs:
            -z: batch of latent samples
        Outputs:
            -x_recon: batch of reconstructed images
        """
        raw_out = self.decoder(z)
        return torch.sigmoid(raw_out)

    def forward(self, x):
        """Do full encode and decode of images
        Inputs:
            - x: batch of images
        Outputs:
            - recon_x: batch of reconstructed images
            - mu: batch of latent mean values
            - logvar: batch of latent logvariances
        """
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, n_samples=1000, encoded=True, **kwargs):
        """
        Generates a sample of data points from the VAE model

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate, by default 1000

        Returns
        -------
        np.ndarray
            Array containing the generated samples
        """
        z = torch.randn(n_samples, self.d)
        samples = (
            self.decode(z).view(n_samples, 28, 28).to(self.device).detach().numpy()
            - 0.5
        ) * 2  # rescale to [-1, 1]
        if encoded:
            samples = torch.tensor(samples).view(n_samples, 1, 28, 28)
            ae = MNISTAutoencoder(d=4).eval()
            ae.load_state_dict(
                torch.load(
                    f"{self.root}/trained_weights/trained_autoencoder_weights.pth",
                    map_location=torch.device("cpu"),
                    weights_only=True,
                )
            )
            samples = ae.encode(samples).detach().numpy()
        return samples


# class GAN:
#     """
#     Source: https://github.com/casey-meehan/data-copying
#     Paper: http://proceedings.mlr.press/v108/meehan20a/meehan20a-supp.pdf
#     """

#     @classmethod
#     def from_pretrained(
#         cls, pretrained_model_name_or_path, cache_dir=None, *inputs, **kwargs
#     ):
#         if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
#             model_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
#             config_file = PRETRAINED_CONFIG_ARCHIVE_MAP[pretrained_model_name_or_path]
#         else:
#             model_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
#             config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)

#         try:
#             resolved_model_file = cached_path(model_file, cache_dir=cache_dir)
#             resolved_config_file = cached_path(config_file, cache_dir=cache_dir)
#         except EnvironmentError:
#             logger.error(
#                 "Wrong model name, should be a valid path to a folder containing "
#                 "a {} file and a {} file or a model name in {}".format(
#                     WEIGHTS_NAME, CONFIG_NAME, PRETRAINED_MODEL_ARCHIVE_MAP.keys()
#                 )
#             )
#             raise

#         logger.info(
#             "loading model {} from cache at {}".format(
#                 pretrained_model_name_or_path, resolved_model_file
#             )
#         )

#         # Load config
#         config = BigGANConfig.from_json_file(resolved_config_file)
#         logger.info("Model config {}".format(config))

#         # Instantiate model.
#         model = cls(config, *inputs, **kwargs)
#         state_dict = torch.load(
#             resolved_model_file,
#             map_location="cpu" if not torch.cuda.is_available() else None,
#         )
#         model.load_state_dict(state_dict, strict=False)
#         return model

#     def __init__(self, config):
#         super(BigGAN, self).__init__()
#         self.config = config
#         self.embeddings = nn.Linear(config.num_classes, config.z_dim, bias=False)
#         self.generator = Generator(config)

#     def forward(self, z, class_label, truncation):
#         assert 0 < truncation <= 1

#         embed = self.embeddings(class_label)
#         cond_vector = torch.cat((z, embed), dim=1)

#         z = self.generator(cond_vector, truncation)
#         return z
