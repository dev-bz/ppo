#include "DebugDraw.h"
#include "box2dtest.h"
#include "imgui.h"
#include <stdlib.h>
#include <string>
#include <vector>
//======
//#include "../main.hpp"
//======
static int state_scroll = 0, save_scroll = 0;
static bool state_show = false;
static int ctrl = 0;
static int touch = 0;
static int running = 1;
static int subStep = 1;
bool inv_motor = false;
bool over = false;
b2Vec2 point(0, 0), old(0, 0);
bool down = false;
char statString[128] = "null";
std::vector<std::string> restore(7, "empty");
#include <env/BoxEnv.hpp>
struct Robot : public testBox::Robot {
  void Draw();
};
Robot robot;
void Robot::Draw() {
  auto c = body->GetPosition();
  c = body->GetWorldPoint(b2Vec2(0, 2.0));
  b2Rot r(act[0] * 0.75f);
  auto d = body->GetWorldVector(0.5f*r.GetYAxis());
  g_debugDraw.DrawSegment(c + d, c - d, b2Color(1, 1, 1, 1));
  d = body->GetWorldPoint(b2Vec2(0.0, 1.6 + act[1]));
  g_debugDraw.DrawSegment(c, d, b2Color(1, 1, 1, 1));

  g_debugDraw.DrawSegment(c, target, b2Color(1, 0.75, 1, 1));
  c = body->GetWorldPoint(b2Vec2(0.6, -1.6));
  d = body->GetWorldPoint(b2Vec2(0.6, -1.6 + shape));
  g_debugDraw.DrawSegment(c, d, b2Color(1, 0.5, 1, 1));
  /*inputs[0] = b2_pi - body->GetAngle();
  while (inputs[0] > b2_pi) inputs[0] -= b2_pi * 2;
  while (inputs[0] < -b2_pi) inputs[0] += b2_pi * 2;
  inputs[0] /= b2_pi;
  inputs[1] = body->GetAngularVelocity() * 0.5;
  for (int i = 0; i < 32 * 32; ++i)
    inputs[i * 3 + 2] = 0.5;
  static std::vector<float> values;
  net.value.getValues(inputs, values);
  static std::vector<float> acts;
  net.net.getValues(inputs, acts);
  int cnt = values.size();
  b2Color a(0, 0, 0, 1);
  b2Color b(1, 1, 1, 1);
  b2Color cc;
  cc.a = 1;
  for (int i = 0; i < cnt; ++i) {
    // value = fmaxf(values[i], value);
    auto l = 0.1 * (values[i]);
    auto r = 1.0f - l;
    cc.r = a.r * r + b.r * l;
    cc.g = a.g * r + b.g * l;
    cc.b = a.b * r + b.b * l;
    setValuesColor(b2Vec2(inputs[i * net.inputSize + 0] + 2.05,
                          inputs[i * net.inputSize + 1]),
                   cc);
    bool t = acts[i * 2 + 0] > 0;
    setValuesColor(
        b2Vec2(inputs[i * net.inputSize + 0], inputs[i * net.inputSize + 1]),
        t ? b2Color(1.0, 1.0f - acts[i * 2 + 0], 1.0 - acts[i * 2 + 0],
                    acts[i * 2 + 1])
          : b2Color(0.0, -acts[i * 2 + 0], 0.0, acts[i * 2 + 1]));
  }
  auto angle = body->GetAngle();
  if (angle > b2_pi)
    angle -= b2_pi + b2_pi;
  else if (angle < -b2_pi)
    angle += b2_pi + b2_pi;
  c.x = fminf(fmaxf(angle / b2_pi, -1.0f), 1.0f);
  c.y = fminf(fmaxf(body->GetAngularVelocity() * 0.5f, -1.0f), 1.0f);
  float tx = 16 + (c.x * 15.5);
  float ty = 16 - (c.y * 15.5);
  g_debugDraw.DrawCircle(0.25 * b2Vec2(tx - 31.5, 15.5 - ty), 0.075,
                         b2Color(1, 1, 1));*/
  static std::vector<float> vals;
  net.value.getValues(net.tuple->obs, vals, 256);
  for (int i = 0; i < 256; ++i) {
    g_debugDraw.DrawSegment(b2Vec2(-4, i * 0.1 + 4.5),
                            b2Vec2(vals[i] * 0.25 - 4, i * 0.1 + 4.5),
                            b2Color(0.25, 1, 0.25));
    g_debugDraw.DrawSegment(
        b2Vec2(-5.5, i * 0.1 + 4.5),
        b2Vec2(net.tuple->rewards[i] * 0.75 - 5.5, i * 0.1 + 4.5),
        net.tuple->dones[i] ? b2Color(0.85, 0.85, 1) : b2Color(1, 0.5, 1));

    g_debugDraw.DrawSegment(
        b2Vec2(-7, i * 0.1 + 4.5),
        b2Vec2(net.tuple->returns[i] * 0.25 - 7, i * 0.1 + 4.5),
        b2Color(1, 0.25, 0.25));
    g_debugDraw.DrawSegment(
        b2Vec2(-8.5, i * 0.1 + 4.5),
        b2Vec2(net.tuple->adv[i] * 0.25 - 8.5, i * 0.1 + 4.5),
        net.tuple->obs[2 * i] > 0
            ? b2Color(1, 1 - net.tuple->obs[2 * i], 1 - net.tuple->obs[2 * i])
            : b2Color(0, -net.tuple->obs[2 * i], 0));
  }
  g_debugDraw.DrawSegment(b2Vec2(-7, +4.5),
                          b2Vec2(-7, net.tuple->position * 0.1 + 4.5),
                          b2Color(1, 1, 1));
  // g_debugDraw.DrawCircle(body->GetPosition() + 4.0 * target, 0.5, b2Color(1,
  // 1, 1, 1));
}
class QueryCallback : public b2QueryCallback {
public:
  QueryCallback(const b2Vec2 &point) {
    m_point = point;
    m_fixture = NULL;
  }
  bool ReportFixture(b2Fixture *fixture) {
    b2Body *body = fixture->GetBody();
    if (body->GetType() == b2_dynamicBody) {
      bool inside = fixture->TestPoint(m_point);
      if (inside) {
        m_fixture = fixture;
        // We are done, terminate the query.
        return false;
      }
    }
    // Continue the query.
    return true;
  }
  b2Vec2 m_point;
  b2Fixture *m_fixture;
};
b2World *w = nullptr;
b2Body *m_groundBody;
b2Vec2 m_mouseWorld;
b2MouseJoint *m_mouseJoint;
void MouseDown(const b2Vec2 &p) {
  m_mouseWorld = g_camera.ConvertScreenToWorld(p);
  if (m_mouseJoint != NULL) {
    return;
  }
  // Make a small box.
  b2AABB aabb;
  b2Vec2 d;
  d.Set(0.001f, 0.001f);
  aabb.lowerBound = m_mouseWorld - d;
  aabb.upperBound = m_mouseWorld + d;
  // Query the world for overlapping shapes.
  QueryCallback callback(m_mouseWorld);
  w->QueryAABB(&callback, aabb);
  if (callback.m_fixture) {
    b2Body *body = callback.m_fixture->GetBody();
    b2MouseJointDef md;
    md.bodyA = m_groundBody;
    md.bodyB = body;
    md.target = m_mouseWorld;
    md.maxForce = 1000.0f * body->GetMass();
    m_mouseJoint = (b2MouseJoint *)w->CreateJoint(&md);
    body->SetAwake(true);
  } else {
    old = p;
    down = true;
  }
}
void MouseUp(const b2Vec2 &p) {
  if (m_mouseJoint) {
    w->DestroyJoint(m_mouseJoint);
    m_mouseJoint = NULL;
  }
  down = false;
}
void MouseMove(const b2Vec2 &p) {
  m_mouseWorld = g_camera.ConvertScreenToWorld(p);
  if (m_mouseJoint) {
    m_mouseJoint->SetTarget(m_mouseWorld);
  } else if (down) {
    b2Vec2 m = m_mouseWorld - g_camera.ConvertScreenToWorld(old);
    g_camera.m_center -= m;
    old = p;
  }
}
float32 angle = 0;
float32 speed = 0;
float32 Kp = 300.0f, Kd = 30.0f;
float time_step = 0.0333f;
float position[3] = {0, 0, 0};
clock_t preTime;
extern class b2Draw *debugDraw;
bool updatePD = true;
bool updateTar = true;
struct BodyInterface {
  virtual int getNumBodies() = 0;
  virtual int getNumJoints() = 0;
  virtual b2Body *GetBody(int part) = 0;
  virtual b2Joint *GetJoint(int part) = 0;
  virtual float32 GetTarget(int part) = 0;
};
// void init_pd(BodyInterface *bx);
// void update_pd(BodyInterface *bx);
struct PDControlller {
  b2RevoluteJoint *joint = nullptr;
  float32 tar = 0.0;
  int nextUpdate = 0;
  int step = 0;
  PDControlller(b2World *w, b2Body *a, b2Body *b) {
    b2RevoluteJointDef j;
    j.Initialize(a, b, a->GetPosition());
    joint = (b2RevoluteJoint *)w->CreateJoint(&j);
    joint->EnableMotor(true);
  }
  void update() {
    if (nextUpdate < step) {
      tar = (drand48() - 0.5) * 6.0;
      nextUpdate = (rand() % 200) + 30;
      step = 0;
    } else
      ++step;
  }
  void updatePD() {
    float turque = (tar - (angle = joint->GetJointAngle())) * Kp -
                   (speed = joint->GetJointSpeed()) * Kd;
    joint->SetMotorSpeed(turque > 0 ? 100.0f : -100.0f);
    joint->SetMaxMotorTorque(b2Abs(turque));
  }
  void draw() {
    auto p = 0.5 * (joint->GetAnchorA() + joint->GetAnchorA());
    {
      b2Rot rot(tar);
      auto r = joint->GetBodyA()->GetWorldVector(rot.GetXAxis());
      g_debugDraw.DrawSegment(p, p + 3.0 * r, b2Color(1, 1, 1));
    }
    {
      float tau = joint->GetMaxMotorTorque() / Kp;
      if (joint->GetMotorSpeed() < 0)
        tau = -tau;
      b2Rot rot(tau + tar);
      auto r = joint->GetBodyA()->GetWorldVector(rot.GetXAxis());
      g_debugDraw.DrawSegment(p, p + 3.0 * r, b2Color(1, 1, 0));
    }
  }
};
struct Body : BodyInterface {
  std::vector<PDControlller *> joints;
  void append(b2Body *a, b2Body *b) {
    joints.push_back(new PDControlller(w, a, b));
  }
  void step() {
    for (auto &joint : joints)
      joint->update();
  }
  void updatePD() {
    for (auto &joint : joints)
      joint->updatePD();
  }
  void draw() {
    for (auto &joint : joints)
      joint->draw();
  }
  void clear() {
    for (auto &joint : joints)
      delete joint;
    joints.clear();
  }
  virtual int getNumBodies() { return joints.size() + 1; }
  virtual int getNumJoints() { return joints.size(); }
  virtual b2Body *GetBody(int part) {
    if (part < joints.size()) {
      return joints[part]->joint->GetBodyA();
    } else
      return joints.back()->joint->GetBodyB();
  }
  virtual b2Joint *GetJoint(int part) {
    if (part <= 0)
      return nullptr;
    return joints[part - 1]->joint;
  }
  virtual float32 GetTarget(int part) {
    if (part <= 0)
      return 0.0f;
    return joints[part - 1]->tar;
  }
};
Body body;
void box2d_init() {
  restore[0] = "save";
  restore[1] = "runThread";
  restore[2] = "saveNetwork";
  restore[3] = "setRunning";
  restore[4] = "syncNetwork";
  if (w)
    return;
  b2Vec2 g(0, -10);
  w = new b2World(g);
  w->SetDebugDraw(debugDraw);
  debugDraw->SetFlags(b2Draw::e_shapeBit /* | b2Draw::e_jointBit*/);
  b2BodyDef def;
  def.type = b2_staticBody;
  m_groundBody = w->CreateBody(&def);
  b2PolygonShape box; // = new b2PolygonShape();
  box.SetAsBox(10.f, .2f, b2Vec2(0, -.2), 0);
  m_groundBody->CreateFixture(&box, 1);
  b2Body *t = m_groundBody;
  box.SetAsBox(.2f, 5.f, b2Vec2(10.2f, 5), 0);
  m_groundBody->CreateFixture(&box, 1);
  box.SetAsBox(.2f, 5.f, b2Vec2(-10.2f, 5), 0);
  m_groundBody->CreateFixture(&box, 1);
  def.type = b2_dynamicBody;
  def.angle = 1;
  box.SetAsBox(.28f, .42f);
  for (int i = 0; i < 20; ++i) {
    def.position.Set(8 * drand48() - 4, 1 + drand48() * 11);
    w->CreateBody(&def)->CreateFixture(&box, 1);
  }
  {
    def.allowSleep = false;
    def.angle = 0.0;
    // def.gravityScale = 0;
    b2Filter filter;
    filter.groupIndex = -1;
#if 0
    box.SetAsBox(1.5f, .42f, b2Vec2(-1.5, 0.0), 0.0);
    def.position.Set(-1.5, 12.0);
    auto *a = w->CreateBody(&def);
    
    a->CreateFixture(&box, 1)->SetFilterData(filter);
    if (1) {
      b2RevoluteJointDef j;
      j.Initialize(m_groundBody, a,
                   b2Vec2(def.position.x - 3.0, def.position.y));
      w->CreateJoint(&j);
    }
    for (int i = 0; i < 3; ++i) {
      def.position.x += 3.0;
      auto *b = w->CreateBody(&def);
      b->CreateFixture(&box, 1)->SetFilterData(filter);
      body.append(a, b);
      a = b;
    }
#endif
    // init_pd(&body);
    {
      box.SetAsBox(0.5f, 1.5f, b2Vec2(0, 1.0f), 0.0f);
      def.gravityScale = 0.0;
      def.position.Set(0.0, 12.0);
      b2Body *a = w->CreateBody(&def);
      a->CreateFixture(&box, 1)->SetFilterData(filter);
      /*a->SetLinearDamping(0.1);
      a->SetAngularDamping(1.0);*/
      if (1) {
        /*b2RevoluteJointDef j;
        j.Initialize(m_groundBody, a, b2Vec2(def.position.x, def.position.y));
        auto b = (b2RevoluteJoint *)w->CreateJoint(&j);*/
        robot.Init(a , "data/tmp"); /*
                 robot.train = false;
                 robot.net.s = 0.1;
                 robot.maxStep = 200;*/
      }
    }
  }
}
void box2d_step() {
  if (running) {
    if (updateTar)
      body.step();
    if (updatePD) {
      // update_pd(&body);
    } else
      body.updatePD();
    w->Step(time_step, 8, 5);
    robot.Step();
  }
}
void box2d_quit() {
  body.clear();
  if (w)
    delete w;
  w = nullptr;
  testBox::setRunning(false);
}
void box2d_gravity(float x, float y) {}
void box2d_draw() {
  body.draw();
  robot.Draw();
  w->DrawDebugData();
}
#define USE_UI
namespace ui {
static const int BUTTON_HEIGHT = 80;
static const int SLIDER_HEIGHT = 80;
static const int SLIDER_MARKER_WIDTH = 40;
static const int CHECK_SIZE = 32;
static const int DEFAULT_SPACING = 16;
static const int TEXT_HEIGHT = 32;
static const int SCROLL_AREA_PADDING = 24;
static const int INDENT_SIZE = 64;
static const int AREA_HEADER = 112;
} // namespace ui
extern float valueAngle;
extern float jointAngle;
int stateSize = 0, inputSize = 0;
void box2d_ui(int width, int height, int mx, int my, unsigned char mbut,
              int scroll) {
  if (mbut) {
    imguiBeginFrame(mx, height - my, mbut, scroll);
    ctrl = 0;
    ++touch;
  } else {
    imguiBeginFrame(0, 0, 0, scroll);
    touch = 0;
    over = false;
    ++ctrl;
  }
  sprintf(statString, "v: %f, j: %f", valueAngle, jointAngle);
#ifdef USE_UI
  char info[260];
  sprintf(info, "(%3d ,%2d)", robot.step, robot.iter);
  int size = (ui::AREA_HEADER + ui::BUTTON_HEIGHT * 3 +
              ui::DEFAULT_SPACING * 2 + ui::SCROLL_AREA_PADDING);
#ifndef DEMO
  size += (ui::BUTTON_HEIGHT + ui::DEFAULT_SPACING) * 4;
#endif
  if (state_show)
    size += ui::BUTTON_HEIGHT * 3 +
            ui::DEFAULT_SPACING * 2; // btMax(width, height) * 3 / 8;
  over |= imguiBeginScrollArea(
      info, width - b2Min(width, height) * 2 / 3 - (width > height ? 120 : 0),
      height - size - 60, b2Min(width, height) * 2 / 3 - 10, size,
      &state_scroll);
  if (imguiButton("Reset", true)) {
    box2d_quit();
    box2d_init();
  }
  if (imguiButton(running ? "Pause" : "Run", true)) {
    running = !running;
  }
/*if (imguiButton("Train", true)) {
        inv_motor = true;
}*/
// imguiLabel("Label");
#ifndef DEMO
  imguiSlider("exp", &robot.net.s, 0.05, 1.0, 0.001, true);
  imguiSlider("zoom", &g_camera.m_zoom, 0.25, 5.0, 0.001, true);
  if (imguiButton(updatePD ? "mutiPD" : "Signle", true)) {
    updatePD = !updatePD;
  }
  if (imguiButton(updateTar ? "Target" : "Wait", true)) {
    updateTar = !updateTar;
  }
#endif
  if (imguiCollapse(statString, 0, state_show, true)) {
    if (state_show) {
      save_scroll = state_scroll;
      state_scroll = 0;
    } else {
      state_scroll = save_scroll;
    }
    state_show = !state_show;
  }
  if (state_show) {
    char info[260];
    for (int i = 0; i < 7; ++i) {
      sprintf(info, "Save at %d: %s", i, restore[i].c_str());
      if (imguiItem(info, true)) {
        switch (i) {
        case 0:
          robot.SaveNet();
          restore[i] = "saved";
          break;
        case 1: {
          int state = testBox::runLearnning(&robot);
          if (state == 2) {
            restore[i] = "runThread sync";
            testBox::syncNetwork(3);
            robot.net.s = 0.05;
          } else if (state == 0) {
            restore[i] = "runThread ok";
            robot.train = false;
            robot.maxStep = 200;
          } else {
            restore[i] = "runThread err";
          }
        } break;
        case 2:
          testBox::saveNetwork();
          restore[i] = "saveNetworked";
          break;
        case 3:
          testBox::setRunning(false);
          restore[i] = "setRunninged";
          break;
        case 4:
          testBox::syncNetwork(0);
          restore[i] = "syncNetworked";
          break;
        }
      }
    }
  }
  imguiEndScrollArea();
  imguiEndFrame();
#endif
  point = b2Vec2(mx, my);
  if (touch == 1) {
    if (!over)
      MouseDown(point);
  } else if (ctrl == 1) {
    MouseUp(point);
  } else {
    MouseMove(point);
  }
}