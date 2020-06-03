import torch
import cyy_pytorch_cpp

tensor = torch.eye(3)
tensor_dict = cyy_pytorch_cpp.data_structure.SyncedSparseTensorDict(
    tensor.to_sparse(), tensor.shape, ""
)
tensor_dict.set_in_memory_number(10)
tensor_dict.enable_debug_logging(False)
tensor_dict.set_saving_thread_number(10)
tensor_dict.set_fetch_thread_number(10)
tensor_dict.set_storage_dir("tensor_dict_dir")
for i in range(10):
    tensor_dict[str(i)] = tensor

tensor_dict.prefetch([str(i) for i in range(100)])
assert str(0) in tensor_dict
for i in tensor_dict.keys():
    assert torch.sum(torch.eq(tensor_dict[i], tensor) == False) == 0
