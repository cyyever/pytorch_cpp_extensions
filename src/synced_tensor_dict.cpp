#include "synced_tensor_dict.hpp"

namespace cyy::pytorch {
  synced_tensor_dict::synced_tensor_dict(const std::string &storage_dir_)
      : storage_dir(storage_dir_) {

    if (storage_dir.exists()) {
      if (!storage_dir.is_dir()) {
        throw std::invalid_argument(storage_dir + " is not a directory");
      }
      for (const auto &f : std::filesystem::directory_iterator(storage_dir)) {
        py::str key = f.path().c_str();
        set_in_disk(key);
      }
    }
  }

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
    auto node = data_info.extra(key);
    if (node.empty()) {
      auto it = data.emplace(data.end(), key, value);
      data_info.emplace(key, {data_state::IN_MEMORY_NEW_DATA, it});
      return;
    }
    auto &[_, it2] = node.value();
    *it2 = value;
    update_access_order(it2);
    data_info.insert(std::move(node));
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
  void update_access_order(decltype(data::iterator) it) {
    data.erase(it);



  }

} // namespace cyy::pytorch
