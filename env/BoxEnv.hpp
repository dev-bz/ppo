#pragma once
#include "Agent.h"
#include <Box2D/Box2D.h>
#include <main.hpp>
#include <stdlib.h>
#include <string>
#include <vector>
#define AGENT_COUNT 12
namespace testBox {
struct Robot {
  Agent *agents[AGENT_COUNT];
  std::vector<std::vector<DataType>> action;
  std::vector<std::vector<DataType>> obs;
  int trains, train, code;
  float32 saved;
  Trainer net;

  std::string model;
  void Init(b2World *w, int type = 0, const char *model = nullptr);
  void Action();
  int Step();
  void Draw();
  void SaveNet();
  void Quit();
};
extern "C" void setRunning(bool b);
extern "C" void syncNetwork(int saveIter = 0);
extern "C" void saveNetwork();
extern "C" int runLearnning(Robot *r = NULL);
} // namespace testBox