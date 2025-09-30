import os
import re
import math
import argparse
from pathlib import Path
from typing import List
import jsonlines
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.data = self.load_data(data_path)

    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def format_triplet_text(triple):
        n = len(triple)
        if n == 3:
            h, r, t = triple
            r = r.replace('_', ' ')
            if r and r[0] == '~':
                r = r[1:]
                return f"{t} {r} {h}"
            return f"{h} {r} {t}"

        if n == 5:
            h, r1, mid, r2, t = triple
            r1 = r1.replace('_', ' ')
            r2 = r2.replace('_', ' ')
            if r1 and r1[0] == '~':
                r1 = r1[1:]
                s1 = f"{mid} {r1} {h}, "
            else:
                s1 = f"{h} {r1} {mid}, "
            if r2 and r2[0] == '~':
                r2 = r2[1:]
                s2 = f"{t} {r2} {mid}"
            else:
                s2 = f"{mid} {r2} {t}"
            return s1 + s2

        if n == 7:
            h, r1, mid1, r2, mid2, r3, t = triple
            r1 = r1.replace('_', ' ')
            r2 = r2.replace('_', ' ')
            r3 = r3.replace('_', ' ')
            if r1 and r1[0] == '~':
                r1 = r1[1:]
                s1 = f"{mid1} {r1} {h}, "
            else:
                s1 = f"{h} {r1} {mid1}, "
            if r2 and r2[0] == '~':
                r2 = r2[1:]
                s2 = f"{mid2} {r2} {mid1}, "
            else:
                s2 = f"{mid1} {r2} {mid2}, "
            if r3 and r3[0] == '~':
                r3 = r3[1:]
                s3 = f"{t} {r3} {mid2}"
            else:
                s3 = f"{mid2} {r3} {t}"
            return s1 + s2 + s3
        raise ValueError(f"Unsupported triplet length: {n}")


    def __getitem__(self, idx):
        sample = self.data[idx]
        pos_list = [self.format_triplet_text(tri) for tri in sample['positive_triplets']]
        neg_text = self.format_triplet_text(sample['negative_triplet'])
        return {
            "question": sample["question"],
            "positive_texts": pos_list,
            "negative_text": neg_text
        }

    @staticmethod
    def load_data(data_path):
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Data not found: {data_path}")
        with jsonlines.open(data_path, 'r') as reader:
            return list(reader)

def collate_fn(batch, tokenizer, max_len=64):
    question_texts= []
    triplet_texts= []
    pos_per_sample = []
    for item in batch:
        q = item["question"]
        pos_list = item["positive_texts"]
        neg = item["negative_text"]
        question_texts.extend([q] * (len(pos_list) + 1))
        triplet_texts.extend(pos_list)
        triplet_texts.append(neg)
        pos_per_sample.append(len(pos_list))
    q_enc = tokenizer(question_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    t_enc = tokenizer(triplet_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return q_enc, t_enc, pos_per_sample

class KGQAEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]  
        return F.normalize(cls, dim=-1)

def info_nce_loss(anchor, positives, negative, temperature):
    pos_sims = F.cosine_similarity(anchor.unsqueeze(0), positives, dim=1)  
    neg_sim = F.cosine_similarity(anchor, negative, dim=0).unsqueeze(0)   
    all_sims = torch.cat([pos_sims, neg_sim], dim=0) / temperature
    m = all_sims.max()
    exps = torch.exp(all_sims - m)
    pos_sum = exps[:-1].sum()
    neg_exp = exps[-1]
    loss = -torch.log(pos_sum / (pos_sum + neg_exp + 1e-12) + 1e-12)
    return loss

def compute_bce_loss(anchor, positives, negative):
    pos_logits = F.cosine_similarity(anchor.unsqueeze(0), positives, dim=1)                  
    neg_logit = F.cosine_similarity(anchor.unsqueeze(0), negative.unsqueeze(0), dim=1)    
    bce = torch.nn.BCEWithLogitsLoss()
    pos_loss = bce(pos_logits, torch.ones_like(pos_logits))
    neg_loss = bce(neg_logit, torch.zeros_like(neg_logit))
    return 0.5 * (pos_loss + neg_loss)

def train_step(q_enc, t_enc, pos_per_sample, model, optimizer, alpha, temperature, scaler, max_grad_norm):
    model.train()
    with torch.amp.autocast('cuda', enabled=(scaler is not None)):
        q_emb = model(q_enc['input_ids'].to(model.device), q_enc['attention_mask'].to(model.device))
        t_emb = model(t_enc['input_ids'].to(model.device), t_enc['attention_mask'].to(model.device))
        losses = []
        i = 0
        for p_num in pos_per_sample:
            total = p_num + 1
            anchor = q_emb[i: i + total][0]
            positives = t_emb[i: i + total][:-1]
            negative = t_emb[i: i + total][-1]
            l_infonce = info_nce_loss(anchor, positives, negative, temperature=temperature)
            l_bce = compute_bce_loss(anchor, positives, negative)
            loss = alpha * l_infonce + (1 - alpha) * l_bce
            losses.append(loss)
            i += total
        loss = torch.stack(losses).mean()

    optimizer.zero_grad(set_to_none=True)
    if scaler is not None:
        scaler.scale(loss).backward()
        if max_grad_norm and max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if max_grad_norm and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

    return loss.item()

@torch.no_grad()
def validate(dataloader, model, alpha, temperature):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for q_enc, t_enc, pos_per_sample in tqdm(dataloader, desc="Valid", leave=False):
        q_emb = model(q_enc['input_ids'].to(model.device), q_enc['attention_mask'].to(model.device))
        t_emb = model(t_enc['input_ids'].to(model.device), t_enc['attention_mask'].to(model.device))
        losses = []
        i = 0
        for p_num in pos_per_sample:
            total = p_num + 1
            anchor = q_emb[i: i + total][0]
            positives = t_emb[i: i + total][:-1]
            negative = t_emb[i: i + total][-1]
            l_infonce = info_nce_loss(anchor, positives, negative, temperature=temperature)
            l_bce = compute_bce_loss(anchor, positives, negative)
            loss = alpha * l_infonce + (1 - alpha) * l_bce
            losses.append(loss)
            i += total
        batch_loss = torch.stack(losses).mean().item()
        total_loss += batch_loss
        n_batches += 1
    return total_loss / max(1, n_batches)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hop', type=int, required=True, choices=[1, 2, 3], help="Number of hops")
    parser.add_argument("--model_name", type=str, default="./sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--save_dir", type=str, default="./checkpoint")
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="If set, load weights from model.pth")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (AMP)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    args = parser.parse_args()
    hop = args.hop
    set_seed(args.seed)
    
    if hop == 1:
        train_path = "./dataset/1-hop-trainset.jsonl"
        valid_path = "./dataset/1-hop-devset.jsonl"
    elif hop == 2:
        train_path = "./dataset/2-hop-trainset.jsonl"
        valid_path = "./dataset/2-hop-devset.jsonl"
    else:
        train_path = "./dataset/3-hop-trainset.jsonl"
        valid_path = "./dataset/3-hop-devset.jsonl"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = str(Path(args.save_dir) / "model.pth")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = CustomDataset(train_path)
    valid_ds = CustomDataset(valid_path)
    collate = lambda batch: collate_fn(batch, tokenizer, max_len=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda"), collate_fn=collate, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"), collate_fn=collate, drop_last=False)
    model = KGQAEncoder(args.model_name).to(device)
    model.device = device
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.amp and torch.cuda.is_available()))
    best_val = math.inf
    
    if args.resume and Path(model_path).exists():
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"[Resume] Loaded weights from '{model_path}'")

    for epoch in range(0, args.epochs):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{args.epochs}"):
            loss = train_step(*batch, model=model, optimizer=optimizer, alpha=args.alpha, temperature=args.temperature, scaler=scaler, max_grad_norm=args.max_grad_norm)
            running += loss
        train_loss = running / max(1, len(train_loader))
        val_loss = validate(valid_loader, model, alpha=args.alpha, temperature=args.temperature)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"[Save] New best ({best_val:.4f}) -> {model_path}")
    print(f"Training end. Best Val Loss: {best_val:.4f} | Saved to {model_path}")

    
if __name__ == "__main__":
    print("Start %d-hop training!" % (hop))
    main()