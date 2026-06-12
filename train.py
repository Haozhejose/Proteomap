import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import argparse
import rdkit
import math, random, sys, os
import numpy as np
from tqdm import tqdm

from fuseprop import *

from joblib import load
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors, DataStructs
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import DataStructs
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mordred import Calculator, descriptors
import warnings
from numpy import VisibleDeprecationWarning

# At the start of your script
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=VisibleDeprecationWarning)

lr_model = load('./Mor+PCtop500_1.joblib')  # 确保文件路径正确

df_pc = pd.read_csv('./PCtop500.csv')
sdf_dir = "./structure"  
sdf_files = {f.split('.')[0]: os.path.join(sdf_dir, f) for f in os.listdir(sdf_dir) if f.endswith(".sdf")}
scaler = load('./scaler.joblib')

df_missindex = pd.read_csv('./missedmordred.csv')
features_to_exclude = df_missindex['ID'].tolist()
# Initialize Mordred calculator with all descriptors
calc = Calculator(descriptors, ignore_3D=True)

mols = []
for mol_name, sdf_path in sdf_files.items():
    suppl = Chem.SDMolSupplier(sdf_path, sanitize=True, removeHs=True)
    for m in suppl:
        if m is not None:
            mols.append(m)

def get_mordred_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    try:    # Calculate all Mordred descriptors
        df_desc = calc(mol)
        df_desc = pd.DataFrame([df_desc.asdict()])
        df_desc = df_desc.astype(float, errors='ignore')  # 让能转换的列都变 float
        df_desc = df_desc.replace([np.inf, -np.inf], np.nan)
        df_desc = df_desc.fillna(0.0)

    #print(desc_series.values)
        # Exclude the features specified in missindex.csv
        filtered_desc = df_desc.drop(features_to_exclude, axis=1, errors='ignore')

        #print(filtered_desc)
        return filtered_desc.values
    finally:
        # 显式清理
        del mol
        if 'df_desc' in locals():
            del df_desc
        if 'filtered_desc' in locals():
            del filtered_desc

def get_morgan_fp(mol, radius=2, nBits=2048):
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)

def descriptor_scoring_with_pc(smiles_list):
    """
    计算分子的target efficiency评分，结合Mordred描述符和protein corona特征。
    通过相似度匹配找到最相似的protein corona数据，组合特征后用机器学习模型预测。
    """
    scores = []
    for query_smiles in smiles_list:
        if query_smiles is None:
            scores.append(0.0)
            continue
        query_mol = Chem.MolFromSmiles(query_smiles)
        query_fp = get_morgan_fp(query_mol)
        query_desc = get_mordred_features(query_smiles)
        
        similarities = []
        for mol in mols:
            sdf_fp = get_morgan_fp(mol)
            if sdf_fp is not None:
                sim = DataStructs.TanimotoSimilarity(query_fp, sdf_fp)
                similarities.append(sim)
            else:
                similarities.append(0.0)

        # 找到最相似的protein corona分子索引
        best_idx = np.argmax(similarities)
        pc_row = df_pc.iloc[best_idx,:].values
        # 组合Mordred特征和protein corona特征
        combined_feature = np.concatenate([query_desc.flatten(), pc_row])
        X = scaler.transform([combined_feature])

        # 用训练好的模型预测target efficiency
        score = lr_model.predict(X)
        scores.append(score[0])

    return scores

def get_reward(smiles_list, scoring_function):
    scores = scoring_function(smiles_list)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)  # 安全处理
    max_score = max(np.max(scores), 1e-6)  # 防止除零
    return torch.as_tensor(scores / max_score, dtype=torch.float32)

#def get_reward(smiles_list, scoring_function):
#    scores = scoring_function(smiles_list)
    # 更有判别力的归一化方式
#    scores_array = np.array(scores, dtype=np.float32).flatten()  # Ensure it's 1D
#    max_score = np.max(scores) if np.max(scores) > 0 else 1  # 防止除以零

    #max_score = max(scores) if max(scores) > 0 else 1  # 防止除以零
#    rewards = scores_array / max_score
    
    #rewards = [score / max_score for score in scores]  # 归一化到 [0, 1]
    #return torch.tensor(rewards, dtype=torch.float)
#    return torch.from_numpy(rewards)

def get_scoring_function(prop_name):
    if prop_name == 'target_efficiency':
        return descriptor_scoring_with_pc
    else:
        raise ValueError(f"Unsupported property name: {prop_name}")

# Decode molecules
def decode_rationales(model, rationale_dataset):
    loader = DataLoader(rationale_dataset, batch_size=40, shuffle=False, num_workers=4, collate_fn=lambda x:x[0])
    model.eval()
    cand_mols = []
    with torch.no_grad():
        #for init_smiles in tqdm(loader, mininterval=600):
        for init_smiles in tqdm(loader):
            final_smiles = model.decode(init_smiles)
            mols = [(x,y) for x,y in zip(init_smiles, final_smiles) if y and '.' not in y and Chem.MolFromSmiles(y) is not None]
            mols = [(x,y) for x,y in mols if Chem.MolFromSmiles(y).HasSubstructMatch(Chem.MolFromSmiles(x))]
            cand_mols.extend(mols)
    return cand_mols

def add_atommap(smiles, num_atoms=1):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for i in range(min(num_atoms, mol.GetNumAtoms())):
        mol.GetAtomWithIdx(i).SetAtomMapNum(1)
    return Chem.MolToSmiles(mol)

def to_float(x):
    return x.item() if isinstance(x, torch.Tensor) else x

# 计算分子的diversity（简单用Tanimoto相似度）
def calc_diversity(smiles_list):
    fps = [Chem.RDKFingerprint(Chem.MolFromSmiles(s)) for s in smiles_list if Chem.MolFromSmiles(s)]
    if len(fps) < 2:
        return 0.0
    sims = []
    for i in range(len(fps)):
        for j in range(i+1, len(fps)):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    if sims:
        return 1 - np.mean(sims)
    else:
        return 0.0

if __name__ == "__main__":
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--proteomap', required=True)
    parser.add_argument('--prop', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--init_model', type=str)
    parser.add_argument('--load_epoch', type=int, default=-1)
    parser.add_argument('--atom_vocab', default=common_atom_vocab)

    parser.add_argument('--rnn_type', type=str, default='LSTM')
    parser.add_argument('--hidden_size', type=int, default=400)
    parser.add_argument('--embed_size', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=40) 
    parser.add_argument('--decode_batch_size', type=int, default=40) #20
    parser.add_argument('--latent_size', type=int, default=20)
    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--diter', type=int, default=3)

    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--clip_norm', type=float, default=20.0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.3)
    parser.add_argument('--num_decode', type=int, default=100) #200

    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--anneal_rate', type=float, default=1.0)
    parser.add_argument('--print_iter', type=int, default=10)
    parser.add_argument('--rl_weight', type=float, default=0.1, help='weight for RL reward loss')

    args = parser.parse_args()
    print(args)

    scoring_function = get_scoring_function(args.prop)

with open(args.proteomap) as f:
    proteomaps = [add_atommap(line.strip()) for line in f if line.strip()]
    proteomap_dataset = SubgraphDataset(proteomaps, args.atom_vocab, args.decode_batch_size, args.num_decode)
        
    model = AtomVGNN(args).cuda()
    if args.load_epoch >= 0:
        path = os.path.join(args.save_dir, f"model.{args.load_epoch}")
        model.load_state_dict(torch.load(path)[1])
    else:
        model.load_state_dict(torch.load(args.init_model))

    print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

    param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
    grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

    history = {
        'loss': [],
        'policy_loss': [],
        'reward_mean': [],
        'target_efficiency': [],
        'diversity': []
    }

    best_target_efficiency = -float('inf')  # 初始设成负无穷，保证任何模型都会比它大

    for epoch in range(args.load_epoch + 1, args.epoch):
        print('epoch', epoch)

        cand_mols = decode_rationales(model, proteomap_dataset)
        model_ckpt = (proteomaps, model.state_dict())
        #torch.save(model_ckpt, os.path.join(args.save_dir, f"model.{epoch}"))

        cand_mols = list(set(cand_mols))
        random.shuffle(cand_mols)

        # Update model
        dataset = MoleculeDataset(cand_mols, args.atom_vocab, args.batch_size)
        dataloader = DataLoader(dataset, batch_size=40, shuffle=True, num_workers=8, collate_fn=lambda x:x[0])
        model.train()

        meters = np.zeros(5)
        for total_step, batch in enumerate(dataset):
            if batch is None: continue

            model.zero_grad()
    
    # 1. 传统reconstruction loss
            loss, kl_div, wacc, tacc, sacc = model(*batch, beta=args.beta)

    # 2. 生成新的分子（policy output）
            batch_pairs = dataset.batches[total_step]
            src_init_smiles = [pair[0] for pair in batch_pairs]
            src_smiles = list(src_init_smiles)

            with torch.no_grad():
                decoded_smiles = model.decode(src_smiles)

    # 3. 计算reward
            rewards = get_reward(decoded_smiles, scoring_function)
            if not torch.is_tensor(rewards):
                rewards = torch.tensor(rewards, device=loss.device).float()
            else:
                rewards = rewards.clone().detach().to(loss.device).float()
            # --- 添加记录reward均值 ---
            avg_reward = rewards.mean().item()
    # 4. policy loss（越高分子越奖励）
            policy_loss = -rewards.mean()  # maximize reward, so minimize -reward

    # 5. 总loss = 重建loss + policy loss
            total_loss = loss + args.rl_weight * policy_loss

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            
# 计算生成分子的平均 target efficiency
        epoch_smiles = []
        with torch.no_grad():
            for (init_smiles, final_smiles) in cand_mols:
                if final_smiles is not None:
                    epoch_smiles.append(final_smiles)

            meters = meters + np.array([to_float(kl_div), to_float(loss), to_float(wacc) * 100, to_float(tacc) * 100, to_float(sacc) * 100])

            if (total_step + 1) % args.print_iter == 0:
                meters /= args.print_iter
                print("[%d] Beta: %.3f, KL: %.2f, loss: %.3f, Word: %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (total_step + 1, args.beta, meters[0], meters[1], meters[2], meters[3], meters[4], param_norm(model), grad_norm(model)))
                sys.stdout.flush()
                meters *= 0

        scheduler.step()

        if epoch_smiles:
            epoch_target_efficiency = np.mean(scoring_function(epoch_smiles))
        else:
            epoch_target_efficiency = 0.0
        
        if (epoch + 1) % 5 == 0:
            model_ckpt = (proteomaps, model.state_dict())
            torch.save(model_ckpt, os.path.join(args.save_dir, f"model_epoch{epoch+1}.pt"))

    # 保存当前最好的模型（根据 target_efficiency）
        if epoch_target_efficiency > best_target_efficiency:
            best_target_efficiency = epoch_target_efficiency
            model_ckpt = (proteomaps, model.state_dict())
            torch.save(model_ckpt, os.path.join(args.save_dir, f"model_best.pt"))


        epoch_diversity = calc_diversity(epoch_smiles)
        num_batches = len(cand_mols)
        history['target_efficiency'].append(epoch_target_efficiency)
        history['loss'].append(total_loss.item()/num_batches)
        history['policy_loss'].append(policy_loss.item()/num_batches)
        history['reward_mean'].append(avg_reward)
        history['diversity'].append(epoch_diversity)

        if (epoch + 1) % 5 == 0:
            fig, axs = plt.subplots(5, 1, figsize=(12, 12))
            axs = axs.flatten()

            metrics = ['loss', 'policy_loss', 'reward_mean', 'target_efficiency', 'diversity']
            titles = ['Total Loss', 'Policy Loss', 'Average Reward', 'Best Target Efficiency', 'Diversity']

            for i, key in enumerate(metrics):
                axs[i].plot(history[key], marker='o')
                axs[i].set_title(titles[i],fontsize=18)
                axs[i].set_xlabel('Epoch', fontsize=16)
                axs[i].set_ylabel(key, fontsize=16)
                axs[i].tick_params(axis='both', which='major', labelsize=14)
                axs[i].yaxis.get_offset_text().set_fontsize(14) 
                axs[i].grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(args.save_dir, f'training_curves{epoch+1}.pdf'))

            df_history = pd.DataFrame(history)
            df_history['epoch'] = np.arange(1, len(history['loss']) + 1)
            df_history.to_csv(os.path.join(args.save_dir, f"training_history_epoch{epoch+1}.csv"), index=False)
