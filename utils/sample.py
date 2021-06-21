import torch

def sample_activations(M, in_activations, layer, n_blocks):
    """
    Sample activations # conv1: ([65536, 576])
    """
    # get indices
    in_features = M.size(1)
    if 'layer1' in layer or 'layer2' in layer:
        n_samples = 1000
    elif 'layer3' in layer or 'layer4' in layer:
        n_samples = 15000
    elif 'fc' in layer or 'classifier' in layer:
        n_samples = 5000
    else:
        return ValueError(layer)
    indices = torch.randint(low=0, high=in_activations.size(0), size=(n_samples // in_features,)).long()
    # sample current in_activations
    in_activations_blocks = in_activations[indices, :].chunk(n_blocks, dim=1)
    return in_activations_blocks