#include "synced_tensor_dict.hpp"

namespace cyy::pytorch {
  torch::Tensor synced_tensor_dict::get(const py::object &key) const {
    auto it = data_info.find(key);
    if (it == data_info.end()) {
      throw py::key_error(py::str(key));
    }
    auto const &[state, it2] = it->second;
    if (state == data_state::IN_MEMORY ||
        state == data_state::IN_MEMORY_NEW_DATA) {
      return *it2;
    }
    // TODO
    throw "unaaa";
  }
  void synced_tensor_dict::set(const py::object &key,
                               const torch::Tensor &value) {
    /* data.emplace(key, value); */
  }
  void synced_tensor_dict::remove(const py::object &key) {
    auto node = data_info.extra(key);
    if (node.empty()) {
      throw py::key_error(py::str(key));
    }
    auto const &[state, it2] = node.value();
    if (it2 != data.end()) {
      data.erase(it2);
    }
  }
} // namespace cyy::pytorch
