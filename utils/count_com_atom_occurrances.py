#!/usr/bin/env python
import torch

com_m = torch.load('exp/subset-d-1m/splits/ranges500-1000000-1000-auto-10000000.3_comweight0/compound_freqs.pt')
atom_m = torch.load('exp/subset-d-1m/splits/ranges500-1000000-1000-auto-10000000.3_comweight0/atom_freqs.pt')

print('Number of compound occurences:\t', int(torch.sum(com_m.to_dense())))
print('Number of atom occurences:\t', int(torch.sum(atom_m.to_dense())))
