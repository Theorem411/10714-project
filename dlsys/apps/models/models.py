import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)

def ResidualConvBN(a1,b1,k1,s1,a2,b2,k2,s2, device=None, dtype="float32"):
    block = nn.Residual(nn.Sequential(
            nn.Conv(in_channels=a1, out_channels=b1, kernel_size=k1, stride=s1, bias=True, device=device, dtype=dtype),
            nn.BatchNorm2d(b1, device=device),
            nn.ReLU(),
            nn.Conv(in_channels=a2, out_channels=b2, kernel_size=k2, stride=s2, bias=True, device=device, dtype=dtype),
            nn.BatchNorm2d(b2, device=device),
            nn.ReLU()
        ))
    return block

def ConvBN(a1,b1,k1,s1,a2,b2,k2,s2, device=None, dtype="float32"):
    block = nn.Sequential(
            nn.Conv(in_channels=a1, out_channels=b1, kernel_size=k1, stride=s1, bias=True, device=device, dtype=dtype),
            nn.BatchNorm2d(b1, device=device),
            # nn.ReLU(),
            # nn.Conv(in_channels=a2, out_channels=b2, kernel_size=k2, stride=s2, bias=True, device=device, dtype=dtype),
            # nn.BatchNorm2d(b2, device=device),
            # nn.ReLU()
        )
    return block
    

class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.model = nn.Sequential(
            ConvBN(3,16,7,4,16,32,3,2,device,dtype),
            # ResidualConvBN(32,32,3,1,32,32,3,1,device,dtype),
            # ConvBN(32,64,3,2,64,128,3,2,device,dtype),
            # ResidualConvBN(128,128,3,1,128,128,3,1,device,dtype),
            # nn.Flatten(),
            # nn.Linear(128, 128, device=device, dtype=dtype),
            # nn.ReLU(),
            # nn.Linear(128, 10, device=device, dtype=dtype), 
        )
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        return self.model.forward(x)
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.device = device
        self.dtype = dtype

        self.embedding = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        self.sequence_model = nn.RNN(embedding_size, hidden_size, num_layers, device=device, dtype=dtype) if seq_model == 'rnn' \
            else nn.LSTM(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)

        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        emb = self.embedding(x) # returns seq_len, batch_size, emb_dim
        out, h = self.sequence_model(emb, h) # returns seq_len, batch_size, hidden_dim
        probs = self.linear(out.reshape((seq_len*bs, self.hidden_size)))
        return probs, h

        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)
