#pragma once

#include "Agent.h"

#define STATE_SIZE 32
#define ACTION_SIZE 4
Agent *createAgent(int id, b2World *w, const b2Vec2 &position,
                   const float32 &angle);