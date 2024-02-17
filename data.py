import re
import torch

def decode(x, chars):
    itoc = {i: c for i, c in enumerate(chars)}
    return ''.join([itoc[i] for i in x])

def encode(x, chars):
    parts = re.split(r'(?<=\\n)|(?<= )|(?=\\n)|(?= )|(?<=\n)|(?=\n)', x)
    ctoi = {c: i for i, c in enumerate(chars)}
    return torch.tensor([ctoi[c] for c in parts if len(c)>0], dtype=torch.long)

def load_data(path):
    with open(path,'r',encoding='utf-8') as f:
        text = f.read()
    
    chars = [']', '\\n', ' ', '\n', '[KEY:', 'CHORDS:', 'COLLECTION:']
    collection = sorted(''.join(re.findall(r'COLLECTION:(.*?)\\n CHORDS:', text)).split(" "))[1:]
    key = re.findall(r'KEY: (.*?)\\n', text)
    chords = re.findall(r'\\n (.*?:)', text)
    digits = [str(i) for i in range(10, 100)]
    chars = sorted(list(set(chars + chords + digits + collection + key)))
    vocab_size = len(chars)

    data = torch.tensor(encode(text, chars), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data, chars, vocab_size, collection

# data loading
def get_batch(train_data, val_data, split, block_size, batch_size, device='cpu'):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
