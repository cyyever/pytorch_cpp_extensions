#include <cyy/cpp_lib/log/log.hpp>
#include <stdexcept>

#include "synced_tensor_dict.hpp"
namespace cyy::pytorch {

  class synced_tensor_dict::save_thread final : public cyy::cxx_lib::runnable {
  public:
    save_thread(synced_tensor_dict &dict_) : dict(dict_) {}

  private:
    void run() override {
      while (true) {
        auto value_opt =
            dict.save_request_queue.pop_front(std::chrono::seconds(1));
        if (!value_opt.has_value()) {
          continue;
        }
        if (!(*value_opt).has_value()) {
          return;
        }
        auto &[key, value, path] = value_opt.value().value();
        try {
          torch::save(value, path.string());
          std::lock_guard lk(dict.data_mutex);
          dict.change_state(key, data_state::SAVING, data_state::IN_DISK);
        } catch (const std::exception &e) {
          LOG_ERROR("torch::save {} failed:{}", path.string(), e.what());
          std::lock_guard lk(dict.data_mutex);
          if (dict.change_state(key, data_state::SAVING,
                                data_state::IN_MEMORY_NEW_DATA)) {
            dict.data.emplace(key, std::move(value));
          }
        }
      }
    }

  private:
    synced_tensor_dict &dict;
  };
} // namespace cyy::pytorch
