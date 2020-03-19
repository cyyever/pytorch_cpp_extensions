import torch
import pytorch_cpp.data_structure

a = pytorch_cpp.data_structure.SyncedTensorDict()
a[5] = torch.ones(1)
# del a[5]
print(a[5])
