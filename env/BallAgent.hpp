#pragma once

#include "Agent.h"
int agentStateSize();
int agentActionSize();
void threadAt(int id);
Agent *createAgent(int id, b2World *w, const b2Vec2 &position, const float32 &angle);
int mirrorAgent(DataType *ob_, DataType *ob, DataType *act, const DataType *_ob_, const DataType *_ob, const DataType *_act);