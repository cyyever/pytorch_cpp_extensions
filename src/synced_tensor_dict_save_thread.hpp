#include <cyy/cpp_lib/log/log.hpp>
#include <stdexcept>

#include "synced_tensor_dict.hpp"
namespace cyy::pytorch {

  class synced_tensor_dict::save_thread final : public cyy::cxx_lib::runnable {
  public:
    save_thread(save_request_queue_type &save_request_queue_,
                save_response_queue_type &save_response_queue_)
        : save_request_queue(save_request_queue_),
          save_response_queue(save_response_queue_) {}
    ~save_thread() override { stop(); }

  private:
    void run() override {
      while (true) {
        auto value_opt = save_request_queue.pop_front(std::chrono::seconds(1));
        if (!value_opt.has_value()) {
          continue;
        }
        if (!(*value_opt).has_value()) {
          return;
        }
        auto const &[key, value, path] = value_opt.value().value();
        try {
          torch::save(value, path.string());
          save_response_queue.emplace_back(key, std::optional<torch::Tensor>{});
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

  class synced_tensor_dict::save_response_thread final
      : public cyy::cxx_lib::runnable {
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
        if (!value) {
          std::lock_guard lk(dict.data_mutex);
          dict.change_state(key, data_state::SAVING, data_state::IN_DISK);
          continue;
        }
        dict.emplace(key, value.value());
      }
    }

  private:
    synced_tensor_dict &dict;
  };
} // namespace cyy::pytorch
