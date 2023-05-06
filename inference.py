import torch
from train import FeedForwardNet, download_mnist_datasets


if __name__ == "__main__":
# load back the model
feed_forward_net = FeedForwardNet()
state_dict = torch.load("feedforwardnet.pth")
feed_forward_net.load_state_dict(state_dict)
# load MNIST validation dataset
# get a sample from the validation dataset
# make an inference