import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import csv

positive_emb_path = "data/positive_embeddings"
negative_emb_path = "data/negative_embeddings"
out_file = "data/dpo_data.csv"
threshold = 0.995
similarity_matches = []

def load_embeddings_in_batches(emb_path, batch_size=2):
    emb_files = [f for f in os.listdir(emb_path) if f.endswith(".pth")]
    total_files = len(emb_files)
    
    for i in range(0, total_files, batch_size):
        batch_files = emb_files[i:i + batch_size]
        batch_dict = {}
        for emb_file in batch_files:
            full_path = os.path.join(emb_path, emb_file)
            data = torch.load(full_path, map_location="cpu")
            batch_dict.update(data)
        yield batch_dict

negative_batches_list = list(load_embeddings_in_batches(negative_emb_path, batch_size=2))
positive_batches = load_embeddings_in_batches(positive_emb_path, batch_size=2)

for pos_batch_idx, pos_emb_dict in enumerate(tqdm(positive_batches, desc="Processing Positive Batches")):
    if not pos_emb_dict:
        continue
    
    pos_seqs, pos_embs = zip(*pos_emb_dict.items())
    pos_embs_tensor = torch.stack([torch.mean(emb.squeeze(0), dim=-1) for emb in pos_embs]) 
    pos_embs_tensor = pos_embs_tensor.to("cuda:0")
    pos_norm = F.normalize(pos_embs_tensor, p=2, dim=1)
    
    for neg_batch_idx, neg_emb_dict in enumerate(tqdm(negative_batches_list, desc=f"Processing Negative Batches for Pos Batch {pos_batch_idx+1}", leave=False)):
        if not neg_emb_dict:
            continue
        
        neg_seqs, neg_embs = zip(*neg_emb_dict.items())
        neg_embs_tensor = torch.stack([torch.mean(emb.squeeze(0), dim=-1) for emb in neg_embs])
        neg_embs_tensor = neg_embs_tensor.to("cuda:0")
        neg_norm = F.normalize(neg_embs_tensor, p=2, dim=1)
        
        cos_sim_matrix = torch.mm(pos_norm, neg_norm.transpose(0, 1))
        pos_indices, neg_indices = torch.where(cos_sim_matrix > threshold)
        
        for pos_idx, neg_idx in zip(pos_indices.tolist(), neg_indices.tolist()):
            pos_seq = pos_seqs[pos_idx]
            neg_seq = neg_seqs[neg_idx]
            similarity = cos_sim_matrix[pos_idx, neg_idx].item()

            similarity_matches.append({
                "positive_seq": pos_seq,
                "negative_seq": neg_seq,
                "similarity": similarity
            })
        
        del neg_embs_tensor, neg_norm, cos_sim_matrix, neg_indices, pos_indices
        torch.cuda.empty_cache()
    
    del pos_embs_tensor, pos_norm, pos_seqs, pos_embs
    torch.cuda.empty_cache()

with open(out_file, "w", newline="") as csvfile:
    fieldnames = ["positive_seq", "negative_seq", "similarity"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for match in similarity_matches:
        writer.writerow(match)

print(f"Similarity matches saved to {out_file}")
print(f"Total matches found: {len(similarity_matches)}")