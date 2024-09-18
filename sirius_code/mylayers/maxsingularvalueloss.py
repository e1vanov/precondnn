import torch
import torch.nn as nn

def circ_mul(C, x):
    """
    Умножение циркулянтной матрицы C на вектор x с использованием FFT.
    
    C: Циркулянтная матрица (представлена первым столбцом)
    x: Вектор для умножения
    """
    # Выполняем умножение с использованием FFT
    result = torch.fft.ifft(torch.fft.fft(C) * torch.fft.fft(x)).to(device='cuda:0').real
    
    return result

def sim_toeplitz_mul(A, x):
    """
    Умножение симметричной матрицы A на вектор x с использованием FFT.
    
    A: Матрица (представлена первой строкой)
    x: Вектор для умножения
    """
    # Преобразуем симметричную матрицу в циркулянтную
    squeezing2 = False
    if len(x.shape) == 1:
        x = x.unsqueeze(0).unsqueeze(0)
        A = A.unsqueeze(0).unsqueeze(0)
        squeezing2 = True
    squeezing1 = False
    if len(x.shape) == 2:
        x = x.unsqueeze(0)
        A = A.unsqueeze(0)
        squeezing1 = True

    C = torch.cat((A, A[:, :,1:].flip(2)), dim=2)
    x_c = torch.cat((x, torch.zeros(x.shape[0], x.shape[1], x.shape[-1]-1)).to(device='cuda:0'), dim=2)
    
    # Выполняем умножение с использованием FFT
    result = circ_mul(C, x_c)[:,:, :x.shape[-1]]
    if squeezing1:
        result = result.squeeze(0)
    if squeezing2:
        result = result.squeeze(0).squeeze(0)
    
    return result

# Класс потерь в виде максимального сингулярного значения
class MaxSingularValueLoss(nn.Module):
    def __init__(self, num_iterations=100):
        super(MaxSingularValueLoss, self).__init__()
        self.num_iterations = num_iterations

    def forward(self, x, circ):
        """
        Находит наибольшее сингулярное число вектора теплицевой A с использованием метода степеней.

        A: Входная матрица
        """
        # Инициализируем случайный вектор
        squeeze1 = False
        if len(x.shape) == 1:
            x = x.unsqueeze(0).unsqueeze(0)
            circ = circ.unsqueeze(0).unsqueeze(0)
            squeeze1 = True
        squeeze2 = False
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            circ = circ.unsqueeze(0)
            squeeze2 = True

        b_k = torch.randn(x.shape).to(device='cuda:0')
        b_k = b_k / torch.norm(b_k, dim=2, keepdim=True)

        for _ in range(self.num_iterations):
            # Умножаем матрицу на вектор
            b_k1 = b_k - sim_toeplitz_mul(x, circ_mul(circ, b_k))

            # Нормализуем вектор
            b_k = b_k1 / torch.norm(b_k1, dim=2, keepdim=True)

        # Наибольшее сингулярное число

        sigma = torch.einsum('bij,bij->bi', b_k, b_k - sim_toeplitz_mul(x, circ_mul(circ, b_k)))

        if squeeze1:
            sigma = sigma.squeeze(0).squeeze(0)
        if squeeze2:
            sigma = sigma.squeeze(0)

        return torch.mean(torch.abs(sigma))
