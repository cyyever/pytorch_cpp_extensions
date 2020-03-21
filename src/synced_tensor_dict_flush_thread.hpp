#include "synced_tensor_dict.hpp"
#include <chrono>
namespace cyy::pytorch {

  class synced_tensor_dict::flush_thread final : public cyy::cxx_lib::runnable {
  public:
    explicit flush_thread(synced_tensor_dict &dict_) : dict(dict_) {}
    ~flush_thread() override { stop(); }

  private:
    void run() override {
      while (true) {
        LOG_INFO("flush begin");
        auto save_tasks = dict.pop_expired_data(true, 10);
        LOG_INFO("flush end");
        if (save_tasks.empty()) {
        LOG_INFO("flush sleep");
          if (wait_stop(std::chrono::milliseconds(1))) {
            return;
          }
          continue;
        }
        LOG_INFO("flush count {}", save_tasks.size());
        dict.flush(save_tasks);
      }
    }

  private:
    synced_tensor_dict &dict;
  };

} // namespace cyy::pytorch
