#pragma once
#include "n.hpp"
struct tExpTuple {
  std::vector<float> states, world;
  std::vector<float> actions;
  std::vector<float> rewards;
  std::vector<bool> dones;
  std::vector<float> adv;
  std::vector<float> values;
  std::vector<float> returns;
  float scale;
  int maxStep;
  int position;
  tExpTuple(int size, int state, int action);
};
struct Trainer {
  int inputSize, outputSize, type;
  Net net, value;
  // float px, py, vx, vy;
  float exp;
  // std::vector<float> v_label;
  std::vector<float> scale;
  // std::vector<float> w;
  // std::vector<float> state;
  std::vector<std::shared_ptr<tExpTuple>> tuples;
  void initTrainer(int s, int a, int type = 0);
  void shutDown();
  const std::vector<float> &preUpdate(const std::vector<float> &input);
  int postUpdate(int id, float shape, const std::vector<float> &o_input,
                 const std::vector<float> &n_input,
                 const std::vector<float> &act, bool done,
                 const std::vector<float> &real);
  void gae(std::vector<float> adv);
  void save(const char *model);
  void load(const char *model);
};
class cMathUtil {
public:
  static double RandDoubleNorm(double mean, double stdev);
  static double EvalGaussian(double mean, double covar, double sample);
  static double EvalGaussianLogp(double mean, double covar, double sample);
};