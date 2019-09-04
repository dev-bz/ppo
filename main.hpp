#pragma once
#include "n.hpp"
struct tExpTuple {
  std::vector<float> states, _states, world;
  std::vector<float> obs;
  std::vector<float> actions;
  std::vector<float> rewards;
  std::vector<bool> dones;
  std::vector<float> adv;
  std::vector<float> values;
  std::vector<float> returns;
  //std::vector<float> logp;
  int maxStep;
  int position;
  tExpTuple(int size, int state, int action);
};
struct Trainer {
  int inputSize, outputSize;
  Net net, value;
  // float px, py, vx, vy;
  float s, exp, save_stddev;
  std::vector<float> v_label;
  std::vector<float> label;
  // std::vector<float> w;
  //std::vector<float> state;
  std::shared_ptr<tExpTuple> tuple,_tuple;
  void initTrainer(const char *model = nullptr);
  const std::vector<float> &preUpdate(const std::vector<float> &input);
  int postUpdate(float shape, const std::vector<float> &o_input,
                 const std::vector<float> &n_input,
                 const std::vector<float> &act, bool done,const std::vector<float>& real);
  void save(const char *model);
};
class cMathUtil {
public:
  static double RandDoubleNorm(double mean, double stdev);
  static double EvalGaussian(double mean, double covar, double sample);
  static double EvalGaussianLogp(double mean, double covar, double sample);
};