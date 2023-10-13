import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.autograd import Variable
import torch.optim as optim

def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def forward_transform(image, info):
    if info["mean"].size() == 3:
        image[:, 0, :, :] = (image[:, 0, :, :] - info["mean"][0]) / info["std"][0]
        image[:, 1, :, :] = (image[:, 1, :, :] - info["mean"][1]) / info["std"][1]
        image[:, 2, :, :] = (image[:, 2, :, :] - info["mean"][2]) / info["std"][2]
    else:
        image[:, 0, :, :] = (image[:, 0, :, :] - info["mean"].squeeze()[0]) / info["std"].squeeze()[0]
        image[:, 1, :, :] = (image[:, 1, :, :] - info["mean"].squeeze()[1]) / info["std"].squeeze()[1]
        image[:, 2, :, :] = (image[:, 2, :, :] - info["mean"].squeeze()[2]) / info["std"].squeeze()[2]
    return image


def clamp_tensor(image, upper_bound, lower_bound):
    image = torch.where(image > upper_bound, upper_bound, image)
    image = torch.where(image < lower_bound, lower_bound, image)
    return image

def get_eps_bounds(eps, x_adv, tensor_std, info):

    pert_epsilon = torch.ones_like(x_adv) * eps / tensor_std
    pert_upper = x_adv + pert_epsilon
    pert_lower = x_adv - pert_epsilon
    upper_bound = torch.ones_like(x_adv)
    lower_bound = torch.zeros_like(x_adv)
    upper_bound = forward_transform(upper_bound, info)
    lower_bound = forward_transform(lower_bound, info)
    upper_bound = torch.min(upper_bound, pert_upper)
    lower_bound = torch.max(lower_bound, pert_lower)

    return upper_bound, lower_bound

# =============================================================================
# Robustness Evaluation
# =============================================================================
def _pgd_whitebox_norm(model,
                  X,
                  y,
                  epsilon,
                  num_steps,
                  step_size,
                  info,
                  random,
                  device
                  ):

    out = model(X)
    out = out[0] if isinstance(out, tuple) else out
    err = (out.data.max(1)[1] != y.data).float().sum()

    std_array = np.asarray(info["std"])
    tensor_std = torch.from_numpy(std_array)
    tensor_std = tensor_std.unsqueeze(0)
    tensor_std = tensor_std.unsqueeze(2)
    tensor_std = tensor_std.unsqueeze(2).float().to(device)

    X_pgd = X.clone() #Variable(X.data, requires_grad=True)
    upper_bound, lower_bound = get_eps_bounds(epsilon, X_pgd, tensor_std, info)

    upper_bound = upper_bound.to(device)
    lower_bound = lower_bound.to(device)
    tensor_std = tensor_std.to(device)
    ones_x = torch.ones_like(X).float().to(device)
    step_size_tensor = ones_x * step_size / tensor_std

    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        random_noise = random_noise / tensor_std
        X_pgd = X_pgd + random_noise
        X_pgd = clamp_tensor(X_pgd, lower_bound, upper_bound)
        X_pgd = Variable(X_pgd.data, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            out = model(X_pgd)
            out = out[0] if isinstance(out, tuple) else out
            loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        eta = step_size_tensor * X_pgd.grad.data.sign()
        X_pgd = X_pgd + eta
        X_pgd = clamp_tensor(X_pgd, upper_bound, lower_bound)
        X_pgd = Variable(X_pgd.data, requires_grad=True)

    out = model(X_pgd)
    out = out[0] if isinstance(out, tuple) else out
    err_pgd = (out.data.max(1)[1] != y.data).float().sum()
    # print('err pgd (white-box): ', err_pgd)
    return err, err_pgd

def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon,
                  num_steps,
                  step_size,
                  random,
                  device
                  ):

    out = model(X)
    out = out[0] if isinstance(out, tuple) else out
    err = (out.data.max(1)[1] != y.data).float().sum()

    X_pgd = Variable(X.data, requires_grad=True)

    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            out = model(X_pgd)
            out = out[0] if isinstance(out, tuple) else out
            loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    out = model(X_pgd)
    out = out[0] if isinstance(out, tuple) else out
    err_pgd = (out.data.max(1)[1] != y.data).float().sum()
    # print('err pgd (white-box): ', err_pgd)
    return err, err_pgd


def eval_adv_robustness(
    model,
    data_loader,
    epsilon,
    num_steps,
    step_size,
    info,
    random=True,
    norm=True,
    device='cuda'
):

    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in tqdm(data_loader, desc='robustness'):
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        if norm:
            err_natural, err_robust = _pgd_whitebox_norm(model, X, y, epsilon, num_steps, step_size, info, random, device)
        else:
            err_natural, err_robust = _pgd_whitebox(model, X, y, epsilon, num_steps, step_size, random, device)

        robust_err_total += err_robust
        natural_err_total += err_natural

    nat_err = natural_err_total.item()
    successful_attacks = robust_err_total.item()
    total_samples = len(data_loader.dataset)

    rob_acc = (total_samples - successful_attacks) / total_samples
    nat_acc = (total_samples - nat_err) / total_samples

    print('=' * 30)
    print(f"Adversarial Robustness = {rob_acc * 100} % ({total_samples - successful_attacks}/{total_samples})")
    print(f"Natural Accuracy = {nat_acc * 100} % ({total_samples - nat_err}/{total_samples})")

    return nat_acc, rob_acc
