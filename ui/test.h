#ifndef _TEST_H_
#define _TEST_H_
#include <Box2D/Box2D.h>
class Test {
  public:
    virtual void init(b2World *w, b2Body *ground) = 0;
    virtual void step(b2World *w) = 0;
    virtual void pre_step(b2World *w) = 0;
    virtual void draw(b2World *w) = 0;
    virtual const char *info() = 0;
    virtual void quit() = 0;
};
#else
#error here
#endif //_TEST_H_