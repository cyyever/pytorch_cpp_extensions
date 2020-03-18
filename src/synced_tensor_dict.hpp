#include <filesystem>
#include <tuple>
#include <unordered_map>

#include <torch/extension.h>

#include "lib/runnable.hpp"

namespace std {
  template <> struct hash<py::object> {
    auto operator()(const py::object &x) const noexcept { return py::hash(x); }
  };
} // namespace std

namespace cyy::pytorch {
  class synced_tensor_dict final {
  public:
    using tensor_list_type = std::list<std::pair<py::object, torch::Tensor>>;
    explicit synced_tensor_dict(const std::string &storage_dir_);
    void release();
    void set(const py::object &key, const torch::Tensor &value);
    torch::Tensor get(const py::object &key) const;
    void remove(const py::object &key);

  private:
    enum class data_state {
      IN_MEMORY,
      IN_MEMORY_NEW_DATA,
      IN_DISK,
      PRE_SAVING,
      SAVING,
      PRE_LOAD,
      LOADING,
    };

  private:
    void set_in_disk(const py::object &key);
    tensor_list_type::iterator
    update_access_order(tensor_list_type::iterator it);
    bool change_state(const py::object &key, data_state old_state,
                      data_state new_state);

  private:
    std::filesystem::path storage_dir;
    tensor_list_type data;
    std::unordered_map<py::object,
                       std::tuple<data_state, tensor_list_type::iterator>>
        data_info;
    bool permenant{false};
  };
} // namespace cyy::pytorch

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  auto sub_m = m.def_submodule("data_structure", "Contains data structures");
  py::class_<cyy::pytorch::synced_tensor_dict>(sub_m, "SyncedTensorDict")
      .def(py::init<const std::string &>())
      .def("__setitem__", &cyy::pytorch::synced_tensor_dict::set)
      .def("__getitem__", &cyy::pytorch::synced_tensor_dict::get)
      .def("__delitem__", &cyy::pytorch::synced_tensor_dict::remove);
}
