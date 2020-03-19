#include <cyy/cpp_lib/log/log.hpp>
#include <stdexcept>

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
        data_info[key] = data_state::IN_DISK;
      }
    }
    save_request_queue.wake_up_on_new_elements = false;
    save_response_queue.wake_up_on_new_elements = false;
    for (size_t i = 0; i < save_thread_num; i++) {
      save_threads.emplace_back(save_request_queue, save_response_queue);
    }
    for (auto &t : save_threads) {
      t.start();
    }
    save_response_threads.emplace_back(*this);
    for (auto &t : save_response_threads) {
      t.start();
    }
    fetch_request_queue.wake_up_on_new_elements = false;
  }

  synced_tensor_dict::~synced_tensor_dict() { release(); }

  void synced_tensor_dict::release() {
    if (permanent) {
      flush_all();
    }
    for (size_t i = 0; i < save_thread_num; i++) {
      save_request_queue.emplace_back();
    }
    for (auto &t : save_threads) {
      t.stop();
    }
    for (auto &t : save_response_threads) {
      t.stop();
    }
    data.clear();
    data_info.clear();

    if (!permanent && !storage_dir.empty()) {
      LOG_DEBUG("remove {}", storage_dir.string());
      std::filesystem::remove_all(storage_dir);
    }
  }

  torch::Tensor synced_tensor_dict::get(const py::object &key) {
    {
      std::lock_guard lk(data_mutex);
      auto it = data_info.find(key);
      if (it == data_info.end()) {
        throw py::key_error(py::str(key));
      }
      auto it2 = data.find(key);
      if (it2 != data.end()) {
        if (it->second != data_state::IN_MEMORY_NEW_DATA) {
          it->second = data_state::IN_MEMORY;
        }
        return *it2;
      }
    }
    prefetch(key);
  }
  void synced_tensor_dict::emplace(const py::object &key,
                                   const torch::Tensor &value) {
    std::lock_guard lk(data_mutex);
    data.emplace(key, value);
    data_info[key] = data_state::IN_MEMORY_NEW_DATA;
  }
  void synced_tensor_dict::erase(const py::object &key) {
    std::lock_guard lk(data_mutex);
    if (!data.erase(key)) {
      throw py::key_error(py::str(key));
    }
    data_info.erase(key);
  }

  bool synced_tensor_dict::flush(bool try_flush) {
    std::unique_lock lk(data_mutex, std::try_to_lock);
    if (!lk.owns_lock()) {
      if (try_flush) {
        return false;
      }
      lk.lock();
    }
    while (data.size() > in_memory_number) {
      auto [key, value] = data.pop_front();
      data_info[key] = data_state::SAVING;
      save_request_queue.emplace_back(
          save_task{key, std::move(value), get_tensor_file_path(key)});
    }
    save_request_queue.wake_up_all_consumers();
    return true;
  }

  void synced_tensor_dict::set_storange_dir(const std::string &storage_dir_) {
    std::lock_guard lk(data_mutex);
    storage_dir = storage_dir_;
    if (!std::filesystem::exists(storage_dir)) {
      std::filesystem::create_directories(storage_dir);
    }
  }

  void synced_tensor_dict::flush_all() {
    auto old_in_memory_number = in_memory_number;
    in_memory_number = 0;
    flush(false);
    in_memory_number = old_in_memory_number;
  }

  std::filesystem::path
  synced_tensor_dict::get_tensor_file_path(py::object key) const {
    if (storage_dir.empty()) {
      throw std::runtime_error("storage_dir is empty");
    }
    return storage_dir / std::filesystem::path(py::str(key));
  }

  void synced_tensor_dict::prefetch(const std::vector<py::object> &keys) {
    for (const auto &key : keys) {
      fetch_request_queue.emplace_back(
          fetch_task{key, get_tensor_file_path(key)});
    }
    fetch_request_queue.wake_up_all_consumers();
  }

  void synced_tensor_dict::prefetch(const py::object &key) {
    fetch_request_queue.emplace_back(
        fetch_task{key, get_tensor_file_path(key)});
    fetch_request_queue.wake_up_all_consumers();
  }

  bool synced_tensor_dict::change_state(const py::object &key,
                                        data_state old_state,
                                        data_state new_state) {
    auto it = data_info.find(key);
    if (it == data_info.end()) {
      return false;
    }
    if (it->second != old_state) {
      return false;
    }
    it->second = new_state;
    return true;
  }

} // namespace cyy::pytorch
