import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader 
from Transformer import GenerationTransformer 
import torch.optim as optim
from BPE.utilities import load_pickle 
from tqdm import tqdm, trange
import logging

from GenerateDataset import GeneratorDataset
from FitModel import fit_generator
from slack_config import slack_urls
from send_slack import send_slack

url = slack_urls().training_in_process()

logging.basicConfig(filename='../logs/training_1.log',level=logging.DEBUG)


vocab_path  = '../data/pickles/vocab'

tokenized_screenplays = load_pickle(f'{vocab_path}/tokenized_screenplays.pkl')

inttostr = load_pickle(f'{vocab_path}/inttostr.pkl')

EPOCHS = 10000
EMBEDDING_DIM = 512
HEADS = 8
DEPTH = 6
SEQ_LEN = 256
BATCH_SIZE = 32
NUM_TOKENS = len(inttostr) + 1

data_set = GeneratorDataset(tokenized_screenplays, SEQ_LEN)

dataloader = DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=150, pin_memory=True)


model = GenerationTransformer(emb = EMBEDDING_DIM, heads = HEADS, depth = DEPTH, seq_length = SEQ_LEN, num_tokens = NUM_TOKENS, device = 'cuda', mask = True, wide = True)

if torch.cuda.is_available():
    model.to("cuda")

optimizer = optim.Adam(model.parameters(), lr = 0.0001)

loss = []
for epoch in trange(1, EPOCHS):
    try:
        l, t = fit_generator(epoch, dataloader, optimizer, model, 100)
        loss.append(l)
        logging.info(t)
        send_slack(url, t)
    except Exception as e: 
        t_ = f'''
            Error at Epoch {epoch}: {e}
        '''
        logging.error(t)
        send_slack(url, t_)

torch.save(model, '../models/final_trained_generator')
send_slack(url, "done with the training")

