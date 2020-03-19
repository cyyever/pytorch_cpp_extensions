#include <cyy/cpp_lib/log/log.hpp>
#include <stdexcept>

#include "synced_tensor_dict.hpp"
namespace cyy::pytorch {

  class synced_tensor_dict::fetch_thread final : public cyy::cxx_lib::runnable {
  public:
    fetch_thread(fetch_request_queue_type &fetch_request_queue_,
                 fetch_response_queue_type &fetch_response_queue_)
        : fetch_request_queue(fetch_request_queue_),
          fetch_response_queue(fetch_response_queue_) {}
    ~fetch_thread() override { stop(); }

  private:
    void run() override {
      while (!needs_stop()) {
        auto value_opt = fetch_request_queue.pop_front(std::chrono::seconds(1));
        if (!value_opt.has_value()) {
          continue;
        }
        auto const &[key, path] = value_opt.value();
        try {
          torch::Tensor value;
          torch::load(value, path.string());
          fetch_response_queue.emplace_back(key, std::move(value));
        } catch (const std::exception &e) {
          LOG_ERROR("torch::load failed:{}", e.what());
          fetch_response_queue.emplace_back(key,
                                            std::optional<torch::Tensor>());
        }
      }
    }

  private:
    fetch_request_queue_type &fetch_request_queue;
    fetch_response_queue_type &fetch_response_queue;
  };

  /* class synced_tensor_dict::fetch_response_thread final */
  /*     : public cyy::cxx_lib::runnable { */
  /* public: */
  /*   fetch_response_thread(synced_tensor_dict &dict_) : dict(dict_) {} */
  /*   ~fetch_response_thread() override { stop(); } */

  /* private: */
  /*   void run() override { */
  /*     while (!needs_stop()) { */
  /*       auto value_opt = */
  /*           dict.fetch_response_queue.pop_front(std::chrono::seconds(1)); */
  /*       if (!value_opt.has_value()) { */
  /*         continue; */
  /*       } */
  /*       auto const &[key, value] = value_opt.value(); */
  /*       if (!value) { */
  /*         std::lock_guard lk(dict.data_mutex); */
  /*         dict.change_state(key, data_state::SAVING, data_state::IN_DISK); */
  /*         continue; */
  /*       } */
  /*       dict.emplace(key, value.value()); */
  /*     } */
  /*   } */

  /* private: */
  /*   synced_tensor_dict &dict; */
  /* }; */
} // namespace cyy::pytorch
