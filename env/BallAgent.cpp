#include "BallAgent.hpp"
#include <ui/DebugDraw.h>
#include <vector>

#define LEG_COUNT 2
#define BODY_COUNT (1 + 4 * LEG_COUNT)

#define STATE_SIZE (BODY_COUNT * 6 - 1 + 5 + 2 * LEG_COUNT) // 62
#define ACTION_SIZE 8
extern int way;
extern float g_time;

// float32 pow_scale[8] = {1.0, 1.0, 0.25, 0.125, 1.0, 1.0, 0.25, 0.125};

struct BodyInterface {
  virtual int getNumBodies() = 0;
  virtual int GetPD() = 0;
  virtual b2Body *GetBody(int part) = 0;
  virtual b2Joint *GetJoint(int part) = 0;
  virtual float32 GetTarget(int part) = 0;
};
void init_pd(BodyInterface *bx);
void update_pd(BodyInterface *bx);
struct PDBody : BodyInterface {
  std::vector<float32> targets;
  std::vector<b2Body *> bodies;
  std::vector<b2RevoluteJoint *> joints;
  static int currentPD;
  int pid;
  void append(b2Body *b, b2RevoluteJoint *j) {
    this->joints.push_back(j);
    this->bodies.push_back(b);
    this->targets.push_back(0.0f);
  }
  void clear() {
    joints.clear();
    bodies.clear();
    targets.clear();
  }
  virtual int getNumBodies() { return this->bodies.size(); }
  virtual b2Body *GetBody(int part) { return this->bodies[part]; }
  virtual b2Joint *GetJoint(int part) { return this->joints[part]; }
  virtual float32 GetTarget(int part) {
    auto j = joints[part];
    if (j) {
      auto scale = (j->GetUpperLimit() - j->GetLowerLimit()) * 0.5;
      auto bias = (j->GetUpperLimit() + j->GetLowerLimit()) * 0.5;
      return targets[part] * scale + bias;
    } else
      return 0.0f;
  }
  virtual int GetPD() { return pid; }
};
int PDBody::currentPD = 0;
void threadAt(int id) { PDBody::currentPD = id; }

struct BallAgent : public Agent {
  b2Body *main;
  b2Body *legs[2 * LEG_COUNT];
  b2Body *foots[2 * LEG_COUNT];
  b2Body *bodies[BODY_COUNT];
  b2RevoluteJoint *joints[BODY_COUNT - 1];
  b2RevoluteJoint *legJoints[2 * LEG_COUNT];
  b2RevoluteJoint *footJoints[2 * LEG_COUNT];
  b2Vec2 m_restitution[2 * LEG_COUNT];
  float32 scale[2];
  float32 bias[2];
  b2Vec2 position, pt, ct;
  float32 power, shape, targetSpeed;
  float32 tv[60], vel[60];
  int tv_index = 0;
  static std::vector<int> target;
  static bool mirror;
  bool testTouch, pd = false, mainDown = true;
  int step = 0, id, iter;
  PDBody body;
  virtual void create(int id, b2World *w, const b2Vec2 &position, const float32 &angle) override {
    if (mirror && target.size() < STATE_SIZE) {
      target.resize(STATE_SIZE);
      for (int i = 0; i < STATE_SIZE; ++i)
        target[i] = i;
      for (int i = 0; i < 12 * LEG_COUNT; ++i)
        target[i + 5] = i + 5 + 12 * LEG_COUNT;
      for (int i = 0; i < LEG_COUNT; ++i)
        target[24 * LEG_COUNT + 5 + i] = 24 * LEG_COUNT + 5 + LEG_COUNT + i;
      // 53 = 55, 54 = 56
    }
    this->position = position;
    this->id = id;
    iter = 0;
    b2BodyDef def;
    def.allowSleep = false;
    def.type = b2_dynamicBody;

    def.angle = angle;
    b2PolygonShape box;
    b2CircleShape circle;
    circle.m_radius = 0.2f;
    circle.m_p.y = -1.5;
    b2Filter mainfilter;
    mainfilter.groupIndex = -1;
    mainfilter.categoryBits = 2;
    b2Filter filter;
    filter.groupIndex = -1;
    filter.maskBits = ~mainfilter.categoryBits;

    b2Fixture *fx;
    for (int i = 0; i < LEG_COUNT; ++i) {
      def.position = position;
      def.position.x += (i - 0.5 * (LEG_COUNT - 1)) * 2;
      box.SetAsBox(0.375, 0.5f, b2Vec2(0, -0.5), 0.0f);
      (legs[i] = w->CreateBody(&def))->CreateFixture(&box, 1.0f)->SetFilterData(filter);
      box.SetAsBox(0.2, 0.75f, b2Vec2(0.0f, -0.75f), 0.0);
      def.position.y -= 1.0f;
      (foots[i] = w->CreateBody(&def))->CreateFixture(&box, 1.0f)->SetFilterData(filter);
      (fx = (foots[i])->CreateFixture(&circle, 0.2f))->SetFilterData(filter);
      fx->SetFriction(10.0f);
    }
    def.position = position;
    box.SetAsBox(1.5f, 1.0f, b2Vec2(0, -0.125f), 0.0f);
    (main = w->CreateBody(&def))->CreateFixture(&box, 1.0f)->SetFilterData(mainfilter);
    for (int i = 0; i < LEG_COUNT; ++i) {
      def.position = position;
      def.position.x += (i - 0.5 * (LEG_COUNT - 1)) * 2;
      box.SetAsBox(0.375, 0.5f, b2Vec2(0, -0.5), 0.0f);
      (legs[i + LEG_COUNT] = w->CreateBody(&def))->CreateFixture(&box, 1.0f)->SetFilterData(filter);
      box.SetAsBox(0.2, 0.75f, b2Vec2(0.0f, -0.75f), 0.0);
      def.position.y -= 1.0f;
      (foots[i + LEG_COUNT] = w->CreateBody(&def))->CreateFixture(&box, 1.0f)->SetFilterData(filter);
      (fx = (foots[i + LEG_COUNT])->CreateFixture(&circle, 0.2f))->SetFilterData(filter);
      fx->SetFriction(10.0f);
    }
    // legs[1]->SetUserData(this);
    // foots[1]->SetUserData(this);
    {
      b2RevoluteJointDef jdef;
      jdef.enableLimit = true;
      jdef.enableMotor = true;
      jdef.lowerAngle = -b2_pi;
      jdef.upperAngle = +0.5f * b2_pi;
      for (int i = 0; i < LEG_COUNT * 2; ++i) {
        jdef.Initialize(main, legs[i], legs[i]->GetPosition());
        legJoints[i] = (b2RevoluteJoint *)w->CreateJoint(&jdef);
      }
      scale[0] = (jdef.upperAngle - jdef.lowerAngle) * 0.5;
      bias[0] = (jdef.upperAngle + jdef.lowerAngle) * 0.5;
    }
    {
      b2RevoluteJointDef jdef;
      jdef.enableLimit = true;
      jdef.enableMotor = true;
      jdef.lowerAngle = 0.25f;
      jdef.upperAngle = 2.85f;
      for (int i = 0; i < LEG_COUNT * 2; ++i) {
        jdef.Initialize(legs[i], foots[i], foots[i]->GetPosition());
        footJoints[i] = (b2RevoluteJoint *)w->CreateJoint(&jdef);
      }
      scale[1] = (jdef.upperAngle - jdef.lowerAngle) * 0.5;
      bias[1] = (jdef.upperAngle + jdef.lowerAngle) * 0.5;
    }
    // state.resize(STATE_SIZE, 0);
    // action.resize(ACTION_SIZE, 0);
    testTouch = true;
    bodies[0] = main;
    /*if (filp) {
            {auto tmp = legs[0]; legs[0] = legs[1]; legs[1] = tmp; }
            {auto tmp = foots[0]; foots[0] = foots[1]; foots[1] = tmp; }
    }*/
    body.append(main, nullptr);
    for (int i = 0; i < LEG_COUNT * 2; ++i) {
      bodies[1 + i * 2] = legs[i];
      bodies[2 + i * 2] = foots[i];

      joints[0 + i * 2] = legJoints[i];
      joints[1 + i * 2] = footJoints[i];
      body.append(legs[i], legJoints[i]);
      body.append(foots[i], footJoints[i]);
    }
    body.pid = PDBody::currentPD;
    init_pd(&body);
    // reset(state.data());
  }
  virtual void reset(DataType *output) override {
    step = 0;
    tv_index = 0;
    auto init_speed = 0; // drand48() * 3;
    auto init_ang = 0;   // (drand48() - 0.5) * 3;
    b2Vec2 rp(main->GetPosition().x, position.y);
    main->SetTransform(rp, 0.0f);
    targetSpeed = 1.0 - (id % 5) / 8.0f;
    for (int i = 0; i < BODY_COUNT; ++i) {
      bodies[i]->SetLinearVelocity(b2Vec2(targetSpeed * 12.0f, 0.0));
      bodies[i]->SetAngularVelocity(init_ang);
      // bodies[i]->GetWorldVector
    }
    if (mainDown)
      for (int i = 0; i < 2; ++i) {
        b2Vec2 p = rp;
        int a = i;
        int b = a + 2;
        if (id % 2) {
          int c = a;
          a = b;
          b = c;
        }
        p.x += (i - 0.5 * (LEG_COUNT - 1)) * 2;
        legs[a]->SetTransform(p, 0.5f);
        foots[a]->SetTransform(legs[a]->GetWorldPoint(b2Vec2(0, -1)), 1.0f);
        legs[b]->SetTransform(p, -1.5f);
        foots[b]->SetTransform(legs[b]->GetWorldPoint(b2Vec2(0, -1)), 0.0f);
      }
    for (int i = 0; i < 2 * LEG_COUNT; ++i) {
      legJoints[i]->SetMaxMotorTorque(0.0f);
      footJoints[i]->SetMaxMotorTorque(0.0f);
    }
    power = 0;
    ++iter;
    update(output, NULL);
  }
  virtual void apply(const DataType *input) override {
    ++step;
    float32 angle, speed;
    float32 Kp = 80.0f, Kd = 10.0f;
    power = 0;
    int ix = -1;

    for (auto &j : body.targets) {
      if (ix >= 0)
        j = // sinf(g_time) * (ix < 4 ? -1.0 : 1.0);
            fmaxf(-1, fminf(input[ix], 1));
      ++ix;
    }
    update_pd(&body);
    ix = 0;
    for (auto j : body.joints) {
      if (j) {
        power += /*pow_scale[ix] */ j->GetMaxMotorTorque();
        ++ix;
      }
    }
#if 0
    for (int j = 0; j < 2 * LEG_COUNT; ++j) {
      int i = /*(legID == 1 && mirror) ? (1 - j) :*/ j;
      {
        auto joint = legJoints[j];
        float turque = (input[i * 2 + 0] * scale[0] + bias[0] - (angle = joint->GetJointAngle())) * Kp - (speed = joint->GetJointSpeed()) * Kd;
        joint->SetMotorSpeed(turque > 0 ? 30.0f : -30.0f);
        joint->SetMaxMotorTorque(b2Abs(turque));
        power += b2Abs(turque) * 0.25;
      }
      {
        auto joint = footJoints[j];
        float turque = (input[i * 2 + 1] * scale[1] + bias[1] - (angle = joint->GetJointAngle())) * Kp - (speed = joint->GetJointSpeed()) * Kd;
        joint->SetMotorSpeed(turque > 0 ? 50.0f : -50.0f);
        joint->SetMaxMotorTorque(b2Abs(turque));
        power += b2Abs(turque);
      }
    }
#endif
  }
  float pows(float x, float p) {
    x = fabsf(x);
    return x - p + p * p / (p + x);
  }
  virtual bool update(DataType *output, DataType *reward) override {
    ct = b2Vec2_zero;
    pt = b2Vec2_zero;
    float32 mass = 0;
    auto root = main->GetPosition();
    int index = 0;
    for (int i = 0; i < BODY_COUNT; ++i) {

      if (i > 0) {
        output[index++] = (bodies[i]->GetWorldCenter().x - root.x) / 2;
        output[index++] = (bodies[i]->GetWorldCenter().y - root.y) / 2;
      } else {
        output[index++] = (root.y - position.y) / 2;
      }
      output[index++] = bodies[i]->GetAngle() / 3;
      output[index++] = bodies[i]->GetAngularVelocity() / 5;
      output[index++] = bodies[i]->GetLinearVelocity().x / 5;
      output[index++] = bodies[i]->GetLinearVelocity().y / 5;
      float32 m = bodies[i]->GetMass();
      mass += m;
      ct += m * bodies[i]->GetLinearVelocity();
      pt += m * bodies[i]->GetWorldCenter();
    }
    ct = 1.0f / mass * ct;
    pt = 1.0f / mass * pt;
    root.y -= position.y;
    // input[ix] = body->GetWorldPoint(b2Vec2(-1.5f, 0)).y / 5;
    // input[ix + 1] = body->GetWorldPoint(b2Vec2(1.5f, 0)).y / 5;
    /*bool touchL = false;
    bool touchR = false;*/
    // float32 Impulse = 0;
    // float32 Impulses[2] = {0, 0};
    int touchIndex = index;
    float loss = 0;
    for (size_t i = 0; i < 2 * LEG_COUNT; i++) {
      output[index] = 0;
      if (testTouch && step > 0)
        for (b2ContactEdge *e = foots[i]->GetContactList(); e; e = e->next) {
          if (e->contact && e->contact->IsTouching()) {
            output[index] = 1.0f;
            loss += foots[i]->GetLinearVelocityFromLocalPoint(b2Vec2(0.0f, -1.5f)).x;
          }
        }
      /*Impulse += b2Abs(output[5 + i * 12]);
      Impulses[i] = b2Abs(output[6 + i * 12]);*/
      // output[index] = footDown[i];
      ++index;
    }
    int targetIndex = index;
    output[index++] = 1.0;
    output[index++] = 0.0;
    output[index++] = 0.0;
    output[index++] = targetSpeed;
    output[index++] = 0.0;
    mainDown = false;
    if (testTouch && step > 0)
      for (b2ContactEdge *e = main->GetContactList(); e; e = e->next) {
        if (e->contact && e->contact->IsTouching()) {
          mainDown = true;
        }
      }
    if (way) {
      if (ct.x > 6.0)
        main->ApplyForce(b2Vec2(5.0f * (targetSpeed * 12.0f - ct.x), 0.0f), pt, false);
    }
    float up = 0.25f;
    float32 tmp = 0.125f - powf((/*v * 8 - sum_vel / t_count*/ output[targetIndex + 3] * 12 - ct.x) * 0.375f * up, 2.0f) - powf((main->GetAngle() + main->GetAngularVelocity() * 0.25) * up * 0.95f, 2.0f) - powf((1.5f - output[targetIndex + 3]) * power * 0.001f * up, 2);
    if (mainDown)
      tmp -= 0.5f;
    for (auto foot : foots)
      tmp -= powf(0.325f * up * (foot->GetWorldPoint(b2Vec2(0.0f, -1.5f)).y - position.y + 1.8f), 2);
    tmp -= powf(loss * up * 0.125f, 2);
    /*int steps = 0;
    if ((output[touchIndex] > 0.5f) && (output[touchIndex + 2] > 0.5)) {
      tmp -= 0.075f;
    }
    if ((output[touchIndex + 1] > 0.5f) && (output[touchIndex + 3] > 0.5)) {
      tmp -= 0.075f;
    }
    if ((output[touchIndex] > 0.5f) || (output[touchIndex + 2] > 0.5)) {
      ++steps;
    }
    if ((output[touchIndex + 1] > 0.5f) || (output[touchIndex + 3] > 0.5)) {
      ++steps;
    }
    if (steps == 2 || steps == 0) {
      tmp += 0.095f;
    }*/
    if (reward) {
      /*float32 v = b2Min(step, 30) * output[targetIndex + 3] / 30.0f;
      tv[tv_index % 30] = v;
      vel[tv_index % 30] = ct.x;
      tv_index = (tv_index + 1);
      if (tv_index >= 120)
        tv_index -= 60;
      float32 sum_tv = 0, sum_vel = 0;
      int t_count = b2Max(1, b2Min(tv_index, 30));
      for (int i = 0; i < t_count; ++i) {
        sum_tv = tv[i] + sum_tv;
        sum_vel = vel[i] + sum_vel;
      }
      v = sum_tv / t_count;*/
      *reward = tmp; // - shape;
    }
    shape = tmp;
    if (pt.x > position.x + 30.0f) {
      for (int i = 0; i < BODY_COUNT; ++i) {
        auto p = bodies[i]->GetPosition();
        p.x -= 40;
        auto r = bodies[i]->GetAngle();
        bodies[i]->SetTransform(p, r);
      }
    } else if (pt.x < position.x - 20.0f) {
      for (int i = 0; i < BODY_COUNT; ++i) {
        auto p = bodies[i]->GetPosition();
        p.x += 40;
        auto r = bodies[i]->GetAngle();
        bodies[i]->SetTransform(p, r);
      }
    }
    // return false;
    return mainDown; // || (id < 3 && step >= 1024);
  }
  virtual void display() override {
    g_debugDraw.DrawSegment(b2Vec2(pt.x, 0), b2Vec2(pt.x, -12 * 0.2 * targetSpeed), b2Color(1, 1, 1));
    g_debugDraw.DrawSegment(b2Vec2(pt.x + 0.1f, 0), b2Vec2(pt.x + 0.1f, -fabsf(ct.x * 0.2f)), b2Color(0.25f, 1.0f, 0.25f));
    for (int i = 0; i < LEG_COUNT + LEG_COUNT; ++i) {
      auto p = foots[i]->GetWorldPoint(b2Vec2(0, -1.5));
      g_debugDraw.DrawSegment(b2Vec2(p.x, -0.1), b2Vec2(p.x, -m_restitution[i].y - 0.1), b2Color(1, 0, 0));
      m_restitution[i].y *= 0.995f;
    }
    if (id == 0) {
      g_debugDraw.DrawCircle(b2Vec2(pt.x, position.y + 1.0f), 0.2f, b2Color(1, 1, 1));
    }
  }
  virtual void destroy() override {}
};
std::vector<int> BallAgent::target;
bool BallAgent::mirror = true;

int agentStateSize() { return STATE_SIZE; }
int agentActionSize() { return ACTION_SIZE; }
Agent *createAgent(int id, b2World *w, const b2Vec2 &position, const float32 &angle) {
  BallAgent *agent = new BallAgent();
  agent->create(id, w, position, angle);
  return agent;
}
int mirrorAgent(DataType *ob_, DataType *ob, DataType *act, const DataType *_ob_, const DataType *_ob, const DataType *_act) {
  if (BallAgent::mirror) {
    for (int i = 0; i < STATE_SIZE; ++i) {
      ob[i] = _ob[i];
      ob_[i] = _ob_[i];
    }
    for (int i = 0; i < ACTION_SIZE; ++i)
      act[i] = _act[(i + 2 * LEG_COUNT) % ACTION_SIZE];
    for (int i = 0; i < STATE_SIZE; ++i) {
      int j = BallAgent::target[i];
      if (j != i) {
        float tmp = ob[i];
        ob[i] = ob[j];
        ob[j] = tmp;
        tmp = ob_[i];
        ob_[i] = ob_[j];
        ob_[j] = tmp;
      }
    }
    return 1;
  }
  return 0;
}