import torch
from nbformat.corpus import words

from vocab import build_vocab,vocab_size
from dataset import build_dataset

def main()->None:
    torch.manual_seed(2147483647)
    words=["emma", "olivia", "ava", "isabella", "sophia"]
    stoi,itos=build_vocab(words)
    X,Y=build_dataset(words,stoi,block_size=3)

    C=torch.randn(vocab_size,2,requires_grad=True)
    emb=C[X]

if __name__=="__main__":
    main()
