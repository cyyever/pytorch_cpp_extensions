#include <filesystem>
#include <torch/extension.h>
#include <tuple>
#include <unordered_map>

#include "lib/runnable.hpp"

namespace std {
  template <> struct hash<py::object> {
    auto operator()(const py::object &x) const noexcept { return py::hash(x); }
  };
} // namespace std

namespace cyy::pytorch {
  class synced_tensor_dict final {
  public:
    using data_iterator_type=decltype(data)::iterator;
    explicit synced_tensor_dict(const std::string &storage_dir_);

    void set(const py::object &key, const torch::Tensor &value);
    torch::Tensor get(const py::object &key) const;
    void remove(const py::object &key);

  private:
    void set_in_disk(const py::object &key);
    void update_access_order(decltype(data::iterator) it);

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
    std::filesystem::path storage_dir;
    std::list<std::pair<py::object, torch::Tensor>> data;
    std::unordered_map<py::object,
                       std::tuple<data_state, decltype(data::iterator)>>
        data_info;
  };
} // namespace cyy::pytorch

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  auto sub_m = m.def_submodule("data_structure", "Contains data structures");
  py::class_<cyy::pytorch::synced_tensor_dict>(sub_m, "SyncedTensorDict")
      .def(py::init<>())
      .def("__setitem__", &cyy::pytorch::synced_tensor_dict::set)
      .def("__getitem__", &cyy::pytorch::synced_tensor_dict::get)
      .def("__delitem__", &cyy::pytorch::synced_tensor_dict::remove);
}
