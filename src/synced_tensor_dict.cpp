#include "synced_tensor_dict.hpp"

namespace cyy::pytorch {
  synced_tensor_dict::synced_tensor_dict(const std::string &storage_dir_)
      : storage_dir(storage_dir_) {

    if (std::filesystem::exists(storage_dir)) {
      if (!std::filesystem::is_directory(storage_dir)) {
        throw std::invalid_argument(storage_dir.string() +
                                    " is not a directory");
      }
      for (const auto &f : std::filesystem::directory_iterator(storage_dir)) {
        py::str key = f.path().c_str();
        set_in_disk(key);
      }
    }
  }

  void synced_tensor_dict::release() {}

  torch::Tensor synced_tensor_dict::get(const py::object &key) const {
    auto it = data_info.find(key);
    if (it == data_info.end()) {
      throw py::key_error(py::str(key));
    }
    auto const &[state, it2] = it->second;
    if (state == data_state::IN_MEMORY ||
        state == data_state::IN_MEMORY_NEW_DATA) {
      return it2->second;
    }
    // TODO
    throw "unaaa";
  }
  void synced_tensor_dict::set(const py::object &key,
                               const torch::Tensor &value) {
    auto node = data_info.extract(key);
    if (node.empty()) {
      auto it = data.emplace(data.end(), key, value);
      data_info.emplace(key, decltype(data_info)::mapped_type{
                                 data_state::IN_MEMORY_NEW_DATA, it});
      return;
    }
    auto &[_, it2] = node.mapped();
    it2->second = value;
    it2 = update_access_order(it2);
    data_info.insert(std::move(node));
  }
  void synced_tensor_dict::remove(const py::object &key) {
    auto node = data_info.extract(key);
    if (node.empty()) {
      throw py::key_error(py::str(key));
    }
    auto const &[state, it2] = node.mapped();
    if (it2 != data.end()) {
      data.erase(it2);
    }
  }
  synced_tensor_dict::tensor_list_type::iterator
  synced_tensor_dict::update_access_order(tensor_list_type::iterator it) {
    auto new_it = data.insert(data.end(), std::move(*it));
    data.erase(it);
    return new_it;
  }

  bool synced_tensor_dict::change_state(const py::object &key,
                                        data_state old_state,
                                        data_state new_state) {
    auto it = data_info.find(key);
    if (it == data_info.end()) {
      return false;
    }
    if (it->second.first != old_state) {
      return false;
    }
    it->second.first = new_state;
    return true;
  }
} // namespace cyy::pytorch
