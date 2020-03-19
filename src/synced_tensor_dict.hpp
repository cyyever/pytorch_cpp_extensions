#pragma once
#include <filesystem>
#include <mutex>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <cyy/cpp_lib/util/ordered_dict.hpp>
#include <cyy/cpp_lib/util/runnable.hpp>
#include <cyy/cpp_lib/util/thread_safe_container.hpp>
#include <torch/extension.h>

namespace std {
  template <> struct hash<py::object> {
    auto operator()(const py::object &x) const noexcept { return py::hash(x); }
  };
} // namespace std

namespace cyy::pytorch {
  class synced_tensor_dict final {
  public:
    explicit synced_tensor_dict(const std::string &storage_dir_);
    ~synced_tensor_dict();
    void release();
    void emplace(const py::object &key, const torch::Tensor &value);
    torch::Tensor get(const py::object &key);
    void erase(const py::object &key);
    void flush_all();
    bool flush(bool try_flush = false);
    void prefetch(const std::vector<py::object> &keys);
    void set_in_memory_number(size_t in_memory_number_) {
      in_memory_number = in_memory_number_;
    }
    void set_storange_dir(const std::string &storage_dir_);

    void set_permanent_storage() { permanent = true; }

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
    using save_task =
        std::tuple<py::object, torch::Tensor, std::filesystem::path>;
    using save_request_queue_type = cyy::cxx_lib::thread_safe_linear_container<
        std::list<std::optional<save_task>>>;
    using save_result = std::pair<py::object, std::optional<torch::Tensor>>;
    using save_response_queue_type =
        cyy::cxx_lib::thread_safe_linear_container<std::list<save_result>>;
    using fetch_task = std::pair<py::object, std::filesystem::path>;
    using fetch_request_queue_type = cyy::cxx_lib::thread_safe_linear_container<
        std::list<std::optional<fetch_task>>>;

    class save_thread final : public cyy::cxx_lib::runnable {
    public:
      save_thread(save_request_queue_type &save_request_queue_,
                  save_response_queue_type &save_response_queue_)
          : save_request_queue(save_request_queue_),
            save_response_queue(save_response_queue_) {}
      ~save_thread() override { stop(); }

    private:
      void run() override {
        while (true) {
          auto value_opt =
              save_request_queue.pop_front(std::chrono::seconds(1));
          if (!value_opt.has_value()) {
            continue;
          }
          if (!(*value_opt).has_value()) {
            return;
          }
          auto const &[key, value, path] = value_opt.value().value();
          try {
            torch::save(value, path.string());
            save_response_queue.emplace_back(key,
                                             std::optional<torch::Tensor>{});
          } catch (const std::exception &e) {
            LOG_ERROR("torch::save failed:{}", e.what());
            save_response_queue.emplace_back(key, std::move(value));
          }
        }
      }

    private:
      save_request_queue_type &save_request_queue;
      save_response_queue_type &save_response_queue;
    };

    class save_response_thread final : public cyy::cxx_lib::runnable {
    public:
      save_response_thread(synced_tensor_dict &dict_) : dict(dict_) {}
      ~save_response_thread() override { stop(); }

    private:
      void run() override {
        while (!needs_stop()) {
          auto value_opt =
              dict.save_response_queue.pop_front(std::chrono::seconds(1));
          if (!value_opt.has_value()) {
            continue;
          }
          auto const &[key, value] = value_opt.value();
        }
      }

    private:
      synced_tensor_dict &dict;
    };

  private:
    bool change_state(const py::object &key, data_state old_state,
                      data_state new_state);
    std::filesystem::path get_tensor_file_path(py::object key) const;

    void prefetch(const py::object &key);

  private:
    std::recursive_mutex data_mutex;
    std::filesystem::path storage_dir;
    cyy::cxx_lib::ordered_dict<py::object, torch::Tensor> data;
    std::unordered_map<py::object, data_state> data_info;
    save_request_queue_type save_request_queue;
    save_response_queue_type save_response_queue;
    size_t save_thread_num{1};
    std::list<save_thread> save_threads;
    std::list<save_response_thread> save_response_threads;
    fetch_request_queue_type fetch_request_queue;
    size_t in_memory_number{128};
    bool permanent{false};
  };
} // namespace cyy::pytorch

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  auto sub_m = m.def_submodule("data_structure", "Contains data structures");
  py::class_<cyy::pytorch::synced_tensor_dict>(sub_m, "SyncedTensorDict")
      .def(py::init<const std::string &>(), py::arg("storage_dir") = "")
      .def("prefetch", (void (cyy::pytorch::synced_tensor_dict::*)(
                           const std::vector<py::object> &keys)) &
                           cyy::pytorch::synced_tensor_dict::prefetch)
      .def("__setitem__", &cyy::pytorch::synced_tensor_dict::emplace)
      .def("__getitem__", &cyy::pytorch::synced_tensor_dict::get)
      .def("__delitem__", &cyy::pytorch::synced_tensor_dict::erase)
      .def("release", &cyy::pytorch::synced_tensor_dict::release)
      .def("flush_all", &cyy::pytorch::synced_tensor_dict::flush_all)
      .def("flush", &cyy::pytorch::synced_tensor_dict::flush,
           py::arg("try_flush") = false);
}
