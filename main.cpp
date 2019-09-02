#include "main.hpp"
#include <conio.h>
#include <env/BoxEnv.hpp>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
tExpTuple::tExpTuple(int size, int state, int action) {
  states.resize(size * state, 0.0f);
  _states.resize(size * state, 0.0f);
  actions.resize(size * action, 0.0f);
  rewards.resize(size, 0.0f);
  returns.resize(size, 0.0f);
  values.resize(size, 0.0f);
  dones.resize(size, false);
  adv.resize(size, 0.0f);
  maxStep = size;
  position = 0;
  // dones[maxStep] = true;
  for (auto &i : states)
    i = drand48() - 0.5;
  obs = states;
  // for (auto &i : returns) i = drand48();
}
void Trainer::initTrainer(const char *model) {
  value.makeNet(inputSize, outputSize, "value.txt", "solver_value.txt");
  v_label.resize(outputSize, 0.25);
  net.makeNet(inputSize, outputSize, "ppo_net.txt", "solver.txt");
  printf("%d,%d\n", inputSize, outputSize);
  label.resize(outputSize, 0.25);
  // w.resize(256 * outputSize, 0.0);
  tuple.reset(new tExpTuple(256, inputSize, outputSize));
  if (model) {
    net.Load(std::string(model) + std::string("_actor.bin"));
    value.Load(std::string(model) + std::string("_critic.bin"));
  }
  s = 0.5;
  // sprintf(statString, "exp: %f", s);
  /*int comp = 0;
  float loss = 1.0;
  do {
          loss = net.trainNet(input, label, w);
          ++comp;
  } while (loss > 0.000002);
  {
          net.syncNet();
          auto v = net.getValue(input);
          for (auto &i : input) printf("%f\t", i);
          printf("\n");
          for (auto &i : v) printf("%f\t", i);
          printf("\nloss= %f, use %d\n", loss, comp);
  }*/
}
const std::vector<float> &Trainer::preUpdate(const std::vector<float> &input) {
  // s = shape;
  //state = input;
  auto &act = net.getValue(input);
  if (s >= 0.05)
    for (auto &i : act)
      i = cMathUtil::RandDoubleNorm(i, s);
  return act;
}
float NormalLogp(float scale, float x) {
  float sx = x / scale;
  return -0.5 * sx * sx - 0.5 * logf(M_PI + M_PI) - logf(scale);
}
float NormalProb(float scale, float x) {
  float sx = x / scale;
  return expf(-0.5 * sx * sx - 0.5 * logf(M_PI + M_PI) - logf(scale));
}
float NormalTD(float scale, float x) {
  // return -x * expf(-0.5 * x * x - 0.5 * logf(M_PI + M_PI) - logf(scale));
  return (-sqrtf(0.5 * M_1_PI) * x * expf(-0.5 * x * x)) /
         (scale * scale * scale);
}
/*float NormalPosition(float scale, float p) {
        return scale * sqrtf(-2.0f * logf(p) - logf(2.0f * M_PI) - 2.0f *
logf(scale));
}*/
int Trainer::postUpdate(float shape, const std::vector<float> &o_input,
                        const std::vector<float> &n_input,
                        const std::vector<float> &act, bool done) {
  // auto o = value.getValue(state)[0];
  // exp = cMathUtil::EvalGaussian(0, 1.0, 0.0);
  for (int i = 0; i < inputSize; ++i) {
    tuple->states[i + tuple->position * inputSize] = o_input[i];
    tuple->_states[i + tuple->position * inputSize] = n_input[i];
  }
  for (int i = 0; i < outputSize; ++i) {
    tuple->actions[i + tuple->position * outputSize] = act[i];
  }
  tuple->rewards[tuple->position] = shape;
  // tuple->values[tuple->position] = o;
  tuple->dones[tuple->position] = done;
  tuple->position = (tuple->position + 1) % tuple->maxStep;
  if (/*tuple->position == 0 ||
        (done && tuple->dones[tuple->maxStep] == false)*/ done) {
    tuple->obs = tuple->states;
    auto &reward = tuple->rewards;
    auto &value = tuple->values;
    auto &_value = tuple->_values;
    auto &done = tuple->dones;
    auto &gaelam = tuple->adv;
    auto &ret = tuple->returns;
    this->value.getValues(tuple->obs, value, 256);
    this->value.getValues(tuple->_states, _value, 256);
    this->net.getValues(tuple->obs, tuple->logp, 256);
    // done[tuple->maxStep] = false;
    /*if (value.size() == tuple->maxStep)
      value.push_back(v);
    else
      value[tuple->maxStep] = v;*/
    const float gamma = 0.975;
    const float lam = 0.935;
    float lastgaelam = 0.0f;
    float total = 0.0;
    float stddev = 0.0;
// reward[tuple->maxStep - 1] = 1.0;
#if 1
    for (int t = (tuple->maxStep + tuple->position - 1) % tuple->maxStep;
         t != tuple->position; t = (tuple->maxStep + t - 1) % tuple->maxStep) {
      float nonterminal = done[t] ? 0.0f : 1.0f;
      // v = reward[t] + gamma * v * nonterminal;
      // float delta = v - value[t];
      float delta = reward[t] + gamma * _value[t] * nonterminal - value[t];
      gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam;
      ret[t] = gaelam[t] + value[t];
      total += gaelam[t];
    }
#else
    //	return_t = np.zeros(path_len)
    // last_val = rewards[-1] + gamma * val_t[-1]
    // return_t[-1] = last_val
    for (int i = tuple->maxStep - 1; i >= 0; --i) {
      auto curr_r = reward[i];
      auto next_ret = i == tuple->maxStep - 1 ? value[i + 1] : ret[i + 1];
      auto curr_val =
          curr_r + gamma * ((1.0 - lam) * value[i + 1] + lam * next_ret);
      ret[i] = curr_val;
    }
#endif
    total /= tuple->maxStep;
    // if (total > 0) total = 0;
    for (int t = tuple->maxStep - 1; t >= 0; --t) {
      gaelam[t] -= total;
      stddev += gaelam[t] * gaelam[t];
    }
    stddev = sqrtf(stddev / tuple->maxStep) + 0.001f;
    for (int t = tuple->maxStep - 1; t >= 0; --t) {
      gaelam[t] /= stddev;
      // gaelam[t] = fmaxf(0.0f, gaelam[t] / stddev);
      /*for (int i = 0; i < outputSize; ++i) {
              auto idx = i + t * outputSize;
              w[idx] = gaelam[t];
              // tuple->logp[idx] = NormalLogp(s, tuple->actions[idx] -
      tuple->logp[idx]);
      }*/
    }
    this->value.LoadTrainData(tuple->obs, tuple->returns, std::vector<float>());
    net.LoadTrainData(tuple->obs, tuple->actions, gaelam);
    std::vector<float> param(4);
    param[0] = s;
    param[1] = 0;
    param[2] = 0.8;
    param[3] = 1.2;
    net.SetTrainParam(param);
    save_stddev = s;
    s = fmaxf(s * 0.998765, 0.05);
    for (int i = 0; i < 10; ++i) {
      this->value.trainNet();
      exp = this->net.trainNet();
    }
    this->value.syncNet();
    this->net.syncNet();
    return 1;
  }
  // s = s - shape;
  return 0;
}
void Trainer::save(const char *model) {
  if (model) {
    net.Save(std::string(model) + std::string("_actor.bin"));
    value.Save(std::string(model) + std::string("_critic.bin"));
  }
}