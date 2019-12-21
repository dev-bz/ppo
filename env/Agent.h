#pragma once
#include <Box2D/Box2D.h>
typedef float DataType;
struct Agent {
  virtual void create(int id, b2World *w, const b2Vec2 &position, const float32 &angle) = 0;
  virtual void reset(DataType *output) = 0;
  virtual void apply(const DataType *input) = 0;
  virtual bool update(DataType *output, DataType *reward) = 0;
  virtual void display() = 0;
  virtual void destroy() = 0;
};