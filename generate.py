import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import math, random, sys
import numpy as np
import argparse
import rdkit
from tqdm import tqdm

from fuseprop import *

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--proteomap', default=None)
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--model', required=True)

parser.add_argument('--num_decode', type=int, default=100)
parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--hidden_size', type=int, default=400)
parser.add_argument('--embed_size', type=int, default=400)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--latent_size', type=int, default=20)
parser.add_argument('--depth', type=int, default=10)
parser.add_argument('--diter', type=int, default=3)

args = parser.parse_args()
random.seed(1)

model = AtomVGNN(args).cuda()
model_ckpt = torch.load(args.model)

if type(model_ckpt) is tuple:
    testdata = model_ckpt[0]
    testdata=unique_rationales(testdata)
    model.load_state_dict(model_ckpt[1])  # 加载模型参数

elif args.proteomap:
    print('loading pre-trained model', file=sys.stderr)
    #testdata = [line.split() for line in open(args.proteomap)] 
    testdata = [line.strip() for line in open(args.proteomap) if line.strip()]
    testdata = unique_rationales(testdata)
    print(testdata)
    model.load_state_dict(model_ckpt)

else:
    raise ValueError("Must provide --proteomap if model does not contain proteomaps")


print('total # rationales:', len(testdata), file=sys.stderr)
model.eval()
dataset = SubgraphDataset(testdata, args.atom_vocab, args.batch_size, args.num_decode)

loader = DataLoader(dataset, batch_size=20, shuffle=False, num_workers=0, collate_fn=lambda x:x[0])
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

with torch.no_grad():
    for init_smiles in tqdm(loader):
        final_smiles = model.decode(init_smiles)
        for x,y in zip(init_smiles, final_smiles):
            print(x, y)

