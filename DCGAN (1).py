import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):# IN
        super(Discriminator,self).__init__()
        self.disc = nn.Sequential(
            # IN : N x channels_img x 64 x 64
            nn.Conv2d(
                channels_img, features_d, kernel_size = 4, stride= 2, padding= 1
            ),# 32x32
            nn.LeakyReLU(0.2),
            self._blok(features_d,features_d*2,4,2,1),# 16 x 16
            self._blok(features_d * 2, features_d * 4, 4, 2, 1),#8 x 8
            self._blok(features_d * 4, features_d * 8, 4, 2, 1),# 4 x 4
            nn.Conv2d(features_d * 8, 1, kernel_size=4,stride=2, padding=0),
            nn.Sigmoid(),
        )
        #blok ha IN
    def _blok(self, in_channels, out_channels,kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias = False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),# segue implementazione paper
        )
    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator,self).__init__()
        self.gen = nn.Sequential(
            #IN: N x z_dim x 1 x 1
            self._blok( z_dim, features_g * 16,4,1,0),#N x f_g*16 x 4 x 4
            self._blok(features_g * 16, features_g * 8, 4, 2, 1),# 8x8
            self._blok(features_g * 8, features_g * 4, 4, 2, 1),#16x16
            self._blok(features_g * 4, features_g * 2, 4, 2, 1),#32x32
            nn.ConvTranspose2d(features_g * 2, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh(), #[-1, 1]
        )
    def _blok(self, in_channels, out_channels,kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias = False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.gen(x)

def initialize_weights(model):
    for n in model.modules():
        if isinstance(n, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(n.weight.data, 0.0, 0.02)

'''
# Definizione del generatore
class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.main = nn.Sequential(
            # IN: N x latent_dim x 1 x 1

            nn.Linear(z_dim, 128 * 8 * 8),  # Aumenta le dimensioni iniziali per adattarle al reshaping
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),  # Unflatten per trasformare il vettore lineare in un tensore 3D
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Raddoppia la dimensione in altezza e larghezza
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Raddoppia nuovamente la dimensione in altezza e larghezza
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # Ultimo strato con 3 canali di output
            nn.Tanh()  # Funzione di attivazione per produrre valori compresi tra -1 e 1  DIM : N x 3 x 64 x 64
        )

    def forward(self, z):
        return self.main(z)


# Definizione del discriminatore
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # IN : N x in_channels x H x W
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


# Funzione per inizializzare i pesi della rete in modo casuale
def initialize_weights(model):
    if isinstance(model, nn.Conv2d) or isinstance(model, nn.ConvTranspose2d) or isinstance(model, nn.Linear):
        nn.init.normal_(model.weight, mean=0.0, std=0.02)
        nn.init.constant_(model.bias, 0.0)
'''


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn(N, in_channels, H, W)
    disc = Discriminator(in_channels, 8)
    initialize_weights(disc)
    #print(disc(x).size)
    assert disc(x).shape == (N,1,1,1)
    gen = Generator(z_dim, in_channels, 8)
    initialize_weights(gen)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W)
    print('Success')

test()