import torch
import torch.optim as optim
import random
import numpy as np


def set_reproducibility(seed=42, deterministic=True):
    # reproducibility stuff
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Note that this Deterministic mode can have a performance impact
    torch.backends.cudnn.deterministic = deterministic  
    print("warn: fix deterministic mode")
    # torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False

def verify_rng(checkpoint):
    recover_rng_state(checkpoint['torch_rng_state'], checkpoint['np_rng_state'])
    assert torch.rand(1) == checkpoint['checksum_pt']
    assert np.random.rand(1) == checkpoint['checksum_np']
    recover_rng_state(checkpoint['torch_rng_state'], checkpoint['np_rng_state'])


def recover_rng_state(torch_rng_state, np_rng_state=None):
    torch.set_rng_state(torch_rng_state)
    if np_rng_state is not None:
        np.random.set_state(np_rng_state)


def save_checkpoint(model: torch.nn.Module, optimizer, epoch, exp_name=None, path=None, other={}):
    chkp_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "np_rng_state": np.random.get_state(),
            }
    if path is None:
        if exp_name is None:
            exp_name = "default_exp"
        path = f"./runs/{exp_name}/checkpoints/checkpoint_{epoch}.pt"
    chkp_dict.update(other)
    torch.save(chkp_dict, path)


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    recover_rng_state(checkpoint['torch_rng_state'], checkpoint['np_rng_state'])



def count_parameters(model: torch.nn.Module) -> int:
  """ Counts the number of trainable parameters of a module
  
  :param model: model that contains the parameters to count
  :returns: the number of parameters in the model
  """
  return sum(p.numel() for p in model.parameters() if p.requires_grad)



def get_model_optimizer(model: torch.nn.Module, opt_type:str, lr:float = None) -> torch.optim.Optimizer:
    """
    Encapsulate the creation of the model's optimizer, to ensure that we use the
    same optimizer everywhere

    :param model: the model that contains the parameter to optimize

    :returns: the model's optimizer
    """
    if lr is None: # defaults that work well
        if opt_type == "Adam":
            lr = 0.001
        elif opt_type == "SGD":
            lr = 0.01
        else:
            lr = 0.001
        
    if opt_type == "Adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    elif opt_type == "SGD":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.1, weight_decay=1e-5)
    elif opt_type == "DAdaptAdam":
        import dadaptation
        return dadaptation.dadapt_adam.DAdaptAdam(model.parameters(), weight_decay=1e-5) #,lr=lr)
    elif opt_type == "DAdaptSGD":
        import dadaptation
        return dadaptation.dadapt_sgd.DAdaptSGD(model.parameters(), momentum=0.1, weight_decay=1e-5)
    elif opt_type == "DAdaptAdaGrad":
        import dadaptation
        return dadaptation.dadapt_adagrad.DAdaptAdaGrad(model.parameters())
    else:
        # default
        #return optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        raise Exception("Unknown Optimizer!")
