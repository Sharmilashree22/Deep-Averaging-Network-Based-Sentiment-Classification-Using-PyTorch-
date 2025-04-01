import torch
import torch.nn as nn
import zipfile
import numpy as np


class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

    def save(self, path):
        """
        Save the model's arguments, vocabulary, and state_dict to a file.

        Args:
            path (str): Path to the file where the model will be saved.
        """
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        """
        Load the model's arguments, vocabulary, and state_dict from a file.

        Args:
            path (str): Path to the file from which the model will be loaded.
        """
        print(f'Loading model from {path}')
        ckpt = torch.load(path)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])

    def load_embedding(vocab, emb_file, emb_size):
        """
        Load pre-trained word embeddings from a file.

        Args:
            vocab: Vocabulary object.
            emb_file (str): Path to the file containing word embeddings.
            emb_size (int): Size of the word embeddings.

        Returns:
            np.ndarray: A 2D NumPy array containing the word embeddings.
        """
        emb = np.zeros((len(vocab), emb_size), dtype=np.float32)
        with open(emb_file, 'r', encoding='utf-8') as file:
            for line in file:
                values = line.strip().split()
                word = values[0]
                if word in vocab.word2idx:
                    emb[vocab.word2idx[word]] = np.array(values[1:], dtype=np.float32)
        return emb


class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.define_model_parameters()
        self.init_model_parameters()

        if args.emb_file is not None:
            self.copy_embedding_from_numpy()

    def define_model_parameters(self):
        """
        Define the model parameters including embedding layer, dropout, and fully connected layer.
        """
        self.embedding = nn.Embedding(len(self.vocab), self.args.emb_size)
        self.dropout = nn.Dropout(self.args.word_drop)
        self.fc = nn.Linear(self.args.emb_size, self.tag_size)

        # Additional code for defining hidden layers if needed
        # Example: self.hidden_layers = nn.ModuleList([nn.Linear(self.args.emb_size, self.args.hid_size) for _ in range(self.args.hid_layer)])

    def init_model_parameters(self):
        """
        Initialize the parameters of the fully connected layer.
        """
        nn.init.uniform_(self.fc.weight, -0.08, 0.08)
        nn.init.zeros_(self.fc.bias)

    def copy_embedding_from_numpy(self):
        """
        Copy pre-trained word embeddings to the model's embedding layer.
        """
        emb_matrix = self.load_embedding(self.vocab, self.args.emb_file, self.args.emb_size)
        self.embedding.weight.data.copy_(torch.from_numpy(emb_matrix))

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor representing word indices.

        Returns:
            torch.Tensor: Output tensor representing predicted scores.
        """
        x_embedded = self.embedding(x)
        x_embedded = self.dropout(x_embedded)
        x_mean = x_embedded.mean(dim=1)
        scores = self.fc(x_mean)
        return scores
