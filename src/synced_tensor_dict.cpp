#include <cyy/cpp_lib/log/log.hpp>
#include <stdexcept>

#include "synced_tensor_dict.hpp"
#include "synced_tensor_dict_fetch_thread.hpp"
#include "synced_tensor_dict_flush_thread.hpp"
#include "synced_tensor_dict_save_thread.hpp"
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
    for (size_t i = 0; i < save_thread_num; i++) {
      save_threads.emplace_back(*this);
    }
    for (auto &t : save_threads) {
      t.start();
    }
    fetch_request_queue.wake_up_on_new_elements = false;
    for (size_t i = 0; i < fetch_thread_num; i++) {
      fetch_threads.emplace_back(*this);
    }
    for (auto &t : fetch_threads) {
      t.start();
    }

    for (size_t i = 0; i < flush_thread_num; i++) {
      flush_threads.emplace_back(*this);
    }
    for (auto &t : flush_threads) {
      t.start();
    }
  }

  synced_tensor_dict::~synced_tensor_dict() { release(); }

  void synced_tensor_dict::release() {
    LOG_INFO("begin release");
    if (permanent) {
      flush_all();
    }
    LOG_INFO("here");
    for (size_t i = 0; i < fetch_thread_num; i++) {
      fetch_request_queue.emplace_back();
    }
    fetch_request_queue.wake_up_all_consumers();
    for (auto &t : fetch_threads) {
      t.stop();
    }
    LOG_INFO("here");
    for (size_t i = 0; i < save_thread_num; i++) {
      save_request_queue.emplace_back();
    }
    save_request_queue.wake_up_all_consumers();
    for (auto &t : save_threads) {
      t.stop();
    }
    LOG_INFO("here");
    for (auto &t : flush_threads) {
      t.stop();
    }
    LOG_INFO("here");
    data.clear();
    data_info.clear();

    if (!permanent && !storage_dir.empty()) {
      LOG_INFO("remove {}", storage_dir.string());
      std::filesystem::remove_all(storage_dir);
    }
  }

  torch::Tensor synced_tensor_dict::get(const py::object &key) {
    while (true) {
      std::unique_lock lk(data_mutex);
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
      if (it->second == data_state::LOAD_FAILED) {
        LOG_ERROR("torch::load {} failed",
                  static_cast<std::string>(py::str(key)));
        throw py::key_error(py::str(key));
      }
      it->second = data_state::PRE_LOAD;
      lk.unlock();
      fetch_request_queue.emplace_back(
          fetch_task{key, get_tensor_file_path(key)});
      fetch_request_queue.wake_up_all_consumers();
      lk.lock();
      new_data_cv.wait(lk);
    }
    throw std::runtime_error("should not be here");
  }
  void synced_tensor_dict::emplace(const py::object &key,
                                   const torch::Tensor &value) {
    {
      std::lock_guard lk(data_mutex);
      data.emplace(key, value);
      data_info[key] = data_state::IN_MEMORY_NEW_DATA;
    }
    flush();
  }
  void synced_tensor_dict::erase(const py::object &key) {
    std::lock_guard lk(data_mutex);
    if (!data.erase(key)) {
      throw py::key_error(py::str(key));
    }
    data_info.erase(key);
  }

  void synced_tensor_dict::flush() {
    auto tasks = pop_expired_data(false, SIZE_MAX);
    flush(tasks);
  }
  void synced_tensor_dict::flush(const std::list<save_task> &tasks) {
    for (auto &task : tasks) {
      save_request_queue.emplace_back(std::move(task));
    }
    if (!tasks.empty()) {
      save_request_queue.wake_up_all_consumers();
    }
  }

  std::list<save_task> synced_tensor_dict::pop_expired_data(bool try_lock,
                                                            size_t max_number) {
    std::list<save_task> expired_data;
    while (expired_data.size() < max_number) {
      std::unique_lock lk(data_mutex, std::try_to_lock);
      if (!lk.owns_lock()) {
        if (try_lock) {
          break;
        }
        lk.lock();
      }

      if (data.size() <= in_memory_number) {
        break;
      }
      std::lock_guard lk(data_mutex);
      auto [key, value] = data.pop_front();
      data_info[key] = data_state::SAVING;
      save_task.emplace_back(
          save_task{key, std::move(value), get_tensor_file_path(key)});
    }
    return expired_data;
  }

  void synced_tensor_dict::set_storage_dir(const std::string &storage_dir_) {
    std::lock_guard lk(data_mutex);
    storage_dir = storage_dir_;
    if (!std::filesystem::exists(storage_dir)) {
      std::filesystem::create_directories(storage_dir);
    }
  }

  void synced_tensor_dict::flush_all() {
    std::lock_guard lk(data_mutex);
    auto old_in_memory_number = in_memory_number;
    in_memory_number = 0;
    flush();
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
    bool flag = false;
    for (const auto &key : keys) {
      {
        std::lock_guard lk(data_mutex);
        auto it = data_info.find(key);
        if (it == data_info.end()) {
          LOG_WARN("skip prefetching {}",
                   static_cast<std::string>(py::str(key)));
          continue;
        }
        if (it->second == data_state::IN_MEMORY ||
            it->second == data_state::IN_MEMORY_NEW_DATA) {
          continue;
        }
        it->second = data_state::PRE_LOAD;
      }
      fetch_request_queue.emplace_back(
          fetch_task{key, get_tensor_file_path(key)});
      flag = true;
    }
    if (flag) {
      fetch_request_queue.wake_up_all_consumers();
    }
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
