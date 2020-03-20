import torch
import pytorch_cpp.data_structure

tensor_dict = pytorch_cpp.data_structure.SyncedTensorDict()
tensor_dict.set_in_memory_number(3)
tensor_dict.set_storage_dir("tensor_dict_dir")
# tensor_dict.set_permanent_storage()
for i in range(100):
    tensor_dict[i] = torch.Tensor([i])
for i in range(100):
    assert tensor_dict[i] == torch.Tensor([i])
print("end")
