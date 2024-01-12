from torch import nn

def create_mlp(input_dim: int, output_dim: int, hidden_dim: int, hidden_layers: int):
    """
    Creates a simple MLP model with the given input and output dimensions, hidden dimensions and number of hidden layers

    Parameters:
    -----------
    `input_dim`: `int`
        input dimension of the model
    `output_dim`: `int`
        output dimension of the model
    `hidden_dim`: `int`
        hidden dimension of the model
    `hidden_layers`: `int`
        number of hidden layers in the model

    Returns:
    --------
    `nn.Module` object
    """

    # this is a bit ugly, but it works
    layers = []
    for _ in range(hidden_layers):
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        input_dim = hidden_dim # update the input dimension for the next layer (this is where it gets ugly)
    layers.append(nn.Linear(input_dim, output_dim))
    
    return nn.Sequential(nn.Flatten(), *layers)