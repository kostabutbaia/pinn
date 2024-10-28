import torch
import numpy as np

def get_all_points(x_space, t_space) -> torch.tensor:
    X, T = np.meshgrid(x_space[1:-1], t_space[1:-1])
    X, T = np.transpose([X.flatten()]), np.transpose([T.flatten()])

    X = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    T = torch.tensor(T, dtype=torch.float32, requires_grad=True)

    return X, T

def split_array_N(arr: list[float], N: int) -> np.ndarray:
    if len(arr) % N != 0:
        raise Exception(f'cannot evenly divide arr with each array of length {N}')
    return np.array([arr[i*N:(i+1)*N] for i in range(int(len(arr)/N))])


def test():
    Nx = 4
    Ny = 4
    x_space = np.linspace(0, 1, Nx)
    y_space = np.linspace(0, 1, Ny)
    torch.tensor([[1] for _ in range(Nx*Ny)])

if __name__ == '__main__':
    test()