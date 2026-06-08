import numpy as np, torch

old = torch.load('/SAN/orengolab/functional-families/janu/contrasted-ff/HUPs_data/HUPs_mmseqs_s90_3.40.50.620/experiments/esm2/HUPs_mmseqs_s90_3.40.50.620_embedded.pt', map_location='cpu')
new = torch.load('/SAN/orengolab/functional-families/janu/contrasted-ff/HUPs_data/HUPs_mmseqs_s90_3.40.50.620/experiments/esm2_corrected/HUPs_s90_esm2_corrected.pt', map_location='cpu')

old_dict = {e['label']: e['mean_representations'][33].float().numpy() for e in old}
new_dict = {e['label']: e['mean_representations'][33].float().numpy() for e in new}

a, b = set(old_dict), set(new_dict)
print('=== Labels ===')
print(f'old:{len(a)} new:{len(b)} shared:{len(a&b)} only_old:{len(a-b)} only_new:{len(b-a)}')

print()
print('=== Embeddings (shared labels) ===')
maes, cos_sims = [], []
for label in a & b:
    u, v = old_dict[label], new_dict[label]
    maes.append(np.abs(u - v).mean())
    cos_sims.append(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))

maes, cos_sims = np.array(maes), np.array(cos_sims)
print(f'MAE    — mean: {maes.mean():.6f}  max: {maes.max():.6f}')
print(f'CosSim — mean: {cos_sims.mean():.6f}  min: {cos_sims.min():.6f}')
print(f'seqs with MAE > 1e-2:      {(maes > 1e-2).sum()}')
print(f'seqs with cos_sim < 0.999: {(cos_sims < 0.999).sum()}')
print(f'seqs with cos_sim < 0.995: {(cos_sims < 0.995).sum()}')
