import torchvision as torchvision
import torchvision.transforms as T
import os 

if __name__ == '__main__':
    os.makedirs(f"data", exist_ok=True)
    torchvision.datasets.MNIST(
        root="data", train=True, transform=T.ToTensor(), download=True)
