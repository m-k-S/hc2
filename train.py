import torch
from model import GPTLanguageModel
from data import get_batch, load_data
import os

@torch.no_grad()
def estimate_loss(train_data, val_data, block_size, batch_size, device, eval_iters, model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(train_data, val_data, 'train', block_size, batch_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

if __name__ == '__main__':
    data_file_path = 'data.txt'
    if not os.path.exists(data_file_path):
        import urllib.request
        print("Downloading data.txt...")
        data_url = 'https://www.dropbox.com/scl/fi/2yod9hatuawm8vjd8aox7/data.txt?rlkey=v530uyk0nrpuz8b66nzc41tmo&dl=1'  # Replace this URL with the actual URL
        urllib.request.urlretrieve(data_url, data_file_path)
        print("Download complete.")


    # model hyperparameters
    block_size = 256
    batch_size = 64 
    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2

    # training parameters
    learning_rate = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 200
    eval_interval = 500
    max_iters = 5000

    torch.manual_seed(1337)

    train_data, val_data, chars, vocab_size, collection = load_data('data.txt')

    # Code to train
    model = GPTLanguageModel(vocab_size, block_size, n_head, n_layer, n_embd, n_head, dropout)
    m = model.to(device)
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(train_data, val_data, block_size, batch_size, device, eval_iters, model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch(train_data, val_data, 'train', block_size, batch_size, device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    # Save the model
    m.eval()
    torch.save(model.state_dict(), 'weights.pth')