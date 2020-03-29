import torch
import cyy_pytorch_cpp

tensor_dict = cyy_pytorch_cpp.data_structure.SyncedTensorDict()
tensor_dict.set_in_memory_number(10)
tensor_dict.set_saving_thread_number(10)
tensor_dict.set_fetch_thread_number(10)
tensor_dict.set_storage_dir("tensor_dict_dir")
# tensor_dict.set_permanent_storage()
for i in range(100):
    tensor_dict[str(i)] = torch.Tensor([i])

tensor_dict.prefetch([str(i) for i in range(100)])
assert str(0) in tensor_dict
for i in range(100):
    assert tensor_dict[str(i)] == torch.Tensor([i])
