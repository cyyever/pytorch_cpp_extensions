#include <chrono>
#include "synced_tensor_dict.hpp"
namespace cyy::pytorch {

  class synced_tensor_dict::flush_thread final : public cyy::cxx_lib::runnable {
  public:
    explicit flush_thread(synced_tensor_dict &dict_) : dict(dict_) {}
    ~flush_thread() override { stop(); }

  private:
    void run() override {
      while (true) {
        auto save_tasks = dict.pop_expired_data(true, 10);
        if (save_tasks.empty()) {
          if (wait_stop(std::chrono::seconds(1))) {
            return;
          }
          continue;
        }
        dict.flush(save_tasks);
      }
    }

  private:
    synced_tensor_dict &dict;
  };

} // namespace cyy::pytorch
