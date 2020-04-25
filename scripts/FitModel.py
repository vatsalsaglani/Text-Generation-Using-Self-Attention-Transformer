import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import os 
import time 
from tqdm import tqdm, trange



# import torch.autograd.Variable as Variable

def fit_generator(epoch, dataloader, optimizer, model, save_every, path = '../models/'):

    start_time = time.time()

    batch_running_loss = 0.0

    for ix, batch in enumerate(tqdm(dataloader)):

        optimizer.zero_grad()

        input, target = batch['input'], batch['output']

        if torch.cuda.is_available():
            input, target = input.cuda(), target.cuda()

        input, target = torch.autograd.Variable(input), torch.autograd.Variable(target)

        output = model(input)

        output = output.transpose(2,1)

        loss = F.nll_loss(output, target, reduction = 'mean')

        batch_running_loss += loss.item()

        loss.backward()
        optimizer.step()

    epoch_loss = batch_running_loss / len(dataloader.dataset)

    t = f"""
    Epoch {epoch}:
        Loss = {epoch_loss}
        Time = {time.time() - start_time}
    """

    if epoch % save_every == 0:

        f_ = os.path.join(path, f'generator_model_{epoch}')
        torch.save(model, f_)

    return epoch_loss, t 