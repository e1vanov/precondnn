import torch
import numpy as np
from sklearn.model_selection import train_test_split

def normal_toeplitz_generator(stds, n_samples):

    for _ in range(n_samples):

        yield np.random.normal(scale=stds)

def generate_symm_toeplitz_ul_dataset(stds=[1.], 
                                      n_samples=10000, 
                                      test_size=0.1,
                                      path='./'):

    X = np.vstack(list(normal_toeplitz_generator(stds, n_samples)))

    X_train, X_test = train_test_split(X, test_size=test_size)

    X_train, X_test = torch.tensor(X_train), torch.tensor(X_test)

    torch.save(X_train, path + 'train.pt')
    torch.save(X_test, path + 'test.pt')
