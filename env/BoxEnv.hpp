#pragma once
#include <Box2D/Box2D.h>
#include <main.hpp>
#include <stdlib.h>
#include <string>
#include <vector>
namespace testBox {
struct Robot {
	b2Body *body;
        float32 angle;
	b2Vec2 position, target;
	int iter, step, maxStep, keepTime;
	int vStep;
	Trainer net;
	float shape,reward;
	bool done, train,filp;
	std::vector<float> input, act,real,old;
	std::string model;
	void Init(b2Body *b, const char *model = nullptr);
	void Action();
	int Step();
	void Update();
	void Reset();
	void Draw();
	void SaveNet();
	//std::vector<float> inputs;
};
extern "C" void setRunning(bool b);
extern "C" void syncNetwork(int saveIter = 0);
extern "C" void saveNetwork();
extern "C" int runLearnning(Robot *r = NULL);
} // namespace testBox