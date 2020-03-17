#include "synced_tensor_dict.hpp"

namespace cyy::pytorch {
  torch::Tensor synced_tensor_dict::get(const py::object &key) const {
    auto it = data.find(key);
    if (it == data.end()) {
      throw py::key_error(py::str(key));
    }
    return it->second;
  }
  void synced_tensor_dict::set(const py::object &key,
                               const torch::Tensor &value) {
    data.emplace(key, value);
  }
  void synced_tensor_dict::remove(const py::object &key) {
    if (data.erase(key) == 0) {
      throw py::key_error(py::str(key));
    }
  }
} // namespace cyy::pytorch
