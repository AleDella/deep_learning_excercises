import torch

from models import LSTM

if __name__ == '__main__':
    # Get the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create a dummy dataset
    toy_dataset = [
        "<bos> ciaone prontone rispondone <eos>",
        "<bos> prontone rispondone <eos>",
        "<bos> prontone rispondone ciaone <eos>",
        "<bos> ciaone <eos>"
    ]
    # Create the dictionaries for conversion
    tok2idx = {
        '<bos>': 0,
        '<eos>': 1,
        'ciaone': 2,
        'prontone': 3,
        'rispondone': 4
    }
    idx2tok = {str(v):k for k,v in tok2idx.items()}
    # Define the components
    vocab_size, hidden_dim = 5, 128
    epochs, lr = 5, 1
    net = LSTM(vocab_size, hidden_dim=hidden_dim, device=device)
    # Initialize the hidden_state and the memory
    hidden = torch.randn((1,vocab_size, hidden_dim))
    memory = torch.randn((1,vocab_size, hidden_dim))
    # Convert the input to numbers
    inpt = toy_dataset[0].split(' ')
    inpt = torch.tensor([[tok2idx[tok] for tok in inpt]])
    # Try the input
    out, (_,_) = net(inpt, hidden, memory)
    print(out.shape)
    
    
    