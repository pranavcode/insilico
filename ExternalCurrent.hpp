#include <vector>

using namespace std;

class ExternalCurrent {
 private:
  vector<double> start_time_vec;
  vector<double> end_time_vec;
  vector<double> volt_vec;
 public:
  void set_ext_currect(const double start_time, const double end_time, const double voltage) {
    start_time_vec.push_back(start_time);
    end_time_vec.push_back(end_time);
    volt_vec.push_back(voltage);
  }
  double I_Ext(const double t) {
    for(vector<double>::size_type iter = 0; iter < start_time_vec.size(); ++iter) {
      if(t >= start_time_vec[iter] && t <= end_time_vec[iter]) {
        return volt_vec[iter];
      }
    }
    return 0;
  }
};
