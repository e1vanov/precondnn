import torch
import numpy as np
from sklearn.model_selection import train_test_split

def normal_toeplitz_generator(stds, n_samples):

    for _ in range(n_samples):

        yield np.random.normal(scale=stds)

def uniform_toeplitz_generator(lows, highs, n_samples):

    for _ in range(n_samples):

        yield np.random.uniform(lows, highs)

def gen_symm_toeplitz_ul_ds_normal(stds=[1.], 
                                   n_samples=10000, 
                                   test_size=0.1,
                                   path='./'):

    X = np.vstack(list(normal_toeplitz_generator(stds, n_samples)))

    X_train, X_test = train_test_split(X, test_size=test_size)

    X_train, X_test = torch.tensor(X_train), torch.tensor(X_test)

    torch.save(X_train, path + 'train.pt')
    torch.save(X_test, path + 'test.pt')

def gen_symm_toeplitz_ul_ds_uniform(lows=[0.],
                                    highs=[1.],
                                    n_samples=10000, 
                                    test_size=0.1,
                                    path='./'):

    X = np.vstack(list(uniform_toeplitz_generator(lows, highs, n_samples)))

    X_train, X_test = train_test_split(X, test_size=test_size)

    X_train, X_test = torch.tensor(X_train), torch.tensor(X_test)

    torch.save(X_train, path + 'train.pt')
    torch.save(X_test, path + 'test.pt')

def gen_symm_band_toeplitz_ul_ds_normal(stds=[1.],
                                        d=32,
                                        n_samples=10000,
                                        test_size=0.1,
                                        path='./'):

    padded_stds = np.zeros(d)
    padded_stds[:len(stds)] = np.array(stds)

    X = np.vstack(list(normal_toeplitz_generator(stds, n_samples)))

    X_train, X_test = train_test_split(X, test_size=test_size)

    X_train, X_test = torch.tensor(X_train), torch.tensor(X_test)

    torch.save(X_train, path + 'train.pt')
    torch.save(X_test, path + 'test.pt')
