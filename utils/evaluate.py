import torch
from torch.autograd import Variable

def evaluate(model, data_loader, device=None, verbose=True):
    # set model mode as test mode.
    model_mode = model.training
    model.train(mode=False)

    total_tested = 0
    total_correct = 0

    for data, labels in data_loader:
        # test the model.
        data = data.to(device)
        labels = labels.to(device)
        scores = model(data)
        scores = scores[0] if isinstance(scores, tuple) else scores
        _, predicted = torch.max(scores, 1)
        # update statistics.
        total_correct += (predicted == labels).sum().item()
        total_tested += len(data)

    # recover the model mode.
    model.train(mode=model_mode)

    # return the precision.
    precision = total_correct / total_tested
    if verbose:
        print('=> precision: {:.3f}'.format(precision))
    return precision
