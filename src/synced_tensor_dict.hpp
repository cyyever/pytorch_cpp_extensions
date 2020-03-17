#include <torch/extension.h>
#include <unordered_map>
/* #include "data_status.hpp" */

#include "lib/runnable.hpp"

namespace std {
  template <> struct hash<py::object> {
    auto operator()(const py::object &x) const noexcept { return py::hash(x); }
  };
} // namespace std

namespace cyy::pytorch {
  class synced_tensor_dict final {
  public:
    synced_tensor_dict() = default;

    void set(const py::object &key, const torch::Tensor &value);
    torch::Tensor get(const py::object &key) const;
    void remove(const py::object &key);

  private:
    std::unordered_map<py::object, torch::Tensor> data;
  };
} // namespace cyy::pytorch

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<cyy::pytorch::synced_tensor_dict>(m, "SyncedTensorDict")
      .def(py::init<>())
      .def("__setitem__", &cyy::pytorch::synced_tensor_dict::set)
      .def("__getitem__", &cyy::pytorch::synced_tensor_dict::get)
      .def("__delitem__", &cyy::pytorch::synced_tensor_dict::remove);
}
