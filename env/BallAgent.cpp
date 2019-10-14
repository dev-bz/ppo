#include "BallAgent.hpp"
#include <ui/DebugDraw.h>
#include <vector>
#define BODY_COUNT 5
struct BallAgent : public Agent {
  b2Body *main;
  b2Body *legs[2];
  b2Body *foots[2];
  b2Body *bodies[BODY_COUNT];
  b2RevoluteJoint *joints[BODY_COUNT - 1];
  b2RevoluteJoint *legJoints[2];
  b2RevoluteJoint *footJoints[2];
  b2Vec2 position, pt, ct;
  float32 footDown[2], plan[2];
  // std::vector<DataType> state;
  // std::vector<DataType> action;
  static std::vector<int> target;
  bool testTouch, mirror = true, pd = false;
  int step = 0, id;
  virtual void create(int id, b2World *w, const b2Vec2 &position,
                      const float32 &angle) override {
    if (mirror && target.size() < STATE_SIZE) {
      target.resize(STATE_SIZE);
      for (int i = 0; i < STATE_SIZE; ++i)
        target[i] = i;
      for (int i = 5; i < 17; ++i)
        target[i] = i + 12;
      target[29] = 30;
    }
    this->position = position;
    this->id = id;
    b2BodyDef def;
    def.allowSleep = false;
    def.type = b2_dynamicBody;

    def.angle = angle;
    b2PolygonShape box;
    b2CircleShape circle;
    circle.m_radius = 0.2f;
    circle.m_p.y = -1.5;
    box.SetAsBox(1.5f, 1.0f);
    b2Filter mainfilter;
    mainfilter.groupIndex = -1;
    mainfilter.categoryBits = 2;
    b2Filter filter;
    filter.groupIndex = -1;
    filter.maskBits = ~mainfilter.categoryBits;

    def.position = position;
    box.SetAsBox(0.375, 0.5f, b2Vec2(0, -0.5), 0.0f);
    (legs[0] = w->CreateBody(&def))
        ->CreateFixture(&box, 1.0f)
        ->SetFilterData(filter);
    box.SetAsBox(0.2, 0.75f, b2Vec2(0.0f, -0.75f), 0.0);
    def.position.y -= 1.0f;
    (foots[0] = w->CreateBody(&def))
        ->CreateFixture(&box, 1.0f)
        ->SetFilterData(filter);
    box.SetAsBox(1.5f, 1.0f, b2Vec2(0, 0.5f), 0.0f);
    def.position = position;
    (main = w->CreateBody(&def))
        ->CreateFixture(&box, 1.0f)
        ->SetFilterData(mainfilter);
    box.SetAsBox(0.375, 0.5f, b2Vec2(0, -0.5), 0.0f);
    (legs[1] = w->CreateBody(&def))
        ->CreateFixture(&box, 1.0f)
        ->SetFilterData(filter);
    box.SetAsBox(0.2, 0.75f, b2Vec2(0.0f, -0.75f), 0.0);
    def.position.y -= 1.0f;
    (foots[1] = w->CreateBody(&def))
        ->CreateFixture(&box, 1.0f)
        ->SetFilterData(filter);
    legs[1]->SetUserData(this);
    foots[1]->SetUserData(this);
    b2Fixture *fx;
    (fx = (foots[0])->CreateFixture(&circle, 0.2f))->SetFilterData(filter);
    fx->SetFriction(10.0f);
    (fx = (foots[1])->CreateFixture(&circle, 0.2f))->SetFilterData(filter);
    fx->SetFriction(10.0f);
    {
      b2RevoluteJointDef jdef;
      jdef.enableLimit = true;
      jdef.enableMotor = true;
      jdef.lowerAngle = -b2_pi;
      jdef.upperAngle = +0.5f * b2_pi;
      jdef.Initialize(main, legs[0], position);
      legJoints[0] = (b2RevoluteJoint *)w->CreateJoint(&jdef);
      jdef.Initialize(main, legs[1], position);
      legJoints[1] = (b2RevoluteJoint *)w->CreateJoint(&jdef);
    }
    {
      b2RevoluteJointDef jdef;
      jdef.enableLimit = true;
      jdef.enableMotor = true;
      jdef.lowerAngle = 0.0f;
      jdef.upperAngle = 2.5f;
      jdef.Initialize(legs[0], foots[0], def.position);
      footJoints[0] = (b2RevoluteJoint *)w->CreateJoint(&jdef);
      jdef.Initialize(legs[1], foots[1], def.position);
      footJoints[1] = (b2RevoluteJoint *)w->CreateJoint(&jdef);
    }
    // state.resize(STATE_SIZE, 0);
    // action.resize(ACTION_SIZE, 0);
    testTouch = true;
    bodies[0] = main;
    /*if (filp) {
            {auto tmp = legs[0]; legs[0] = legs[1]; legs[1] = tmp; }
            {auto tmp = foots[0]; foots[0] = foots[1]; foots[1] = tmp; }
    }*/
    bodies[1] = legs[0];
    bodies[2] = foots[0];
    bodies[3] = legs[1];
    bodies[4] = foots[1];

    joints[0] = legJoints[0];
    joints[1] = footJoints[0];
    joints[2] = legJoints[1];
    joints[3] = footJoints[1];
    // reset(state.data());
    if (id == 0) {
      for (int i = 0; i < BODY_COUNT; ++i) {
        printf("MASS [%d] %f\n", i, bodies[i]->GetMass());
      }
    }
  }
  virtual void reset(DataType *output) override {
    step = 0;
    auto init_speed = drand48() * 3;
    auto init_ang = (drand48() - 0.5) * 3;
    for (int i = 0; i < BODY_COUNT; ++i) {
      bodies[i]->SetTransform(position, 0.0f);
      bodies[i]->SetLinearVelocity(b2Vec2(init_speed, 0.0));
      bodies[i]->SetAngularVelocity(init_ang);
      // bodies[i]->GetWorldVector
    }
    legs[id % 2]->SetTransform(position, 0.5f);
    legs[1 - (id % 2)]->SetTransform(position, -1.5f);
    foots[id % 2]->SetTransform(legs[id % 2]->GetWorldPoint(b2Vec2(0, -1)),
                                1.0f);
    foots[1 - (id % 2)]->SetTransform(
        legs[1 - (id % 2)]->GetWorldPoint(b2Vec2(0, -1)), 0.0f);

    for (int i = 0; i < 2; ++i) {
      legJoints[i]->SetMaxMotorTorque(0.0f);
      footJoints[i]->SetMaxMotorTorque(0.0f);
    }
    footDown[id % 2] = 0.0f;
    footDown[1 - (id % 2)] = 1.0f;
    plan[0] = footDown[0];
    plan[1] = footDown[1];
    update(output, NULL);
  }
  virtual void apply(const DataType *input) override {
    ++step;
    float32 angle, speed;
    float32 Kp = 80.0f, Kd = 10.0f;
    int legID = plan[0] > plan[1] ? 0 : 1;
    for (int j = 0; j < 2; ++j) {
      int i = (legID == 1 && mirror) ? (1 - j) : j;
      {
        auto joint = legJoints[j];
        float turque =
            (input[i * 2 + 0] - (angle = joint->GetJointAngle())) * Kp -
            (speed = joint->GetJointSpeed()) * Kd;
        joint->SetMotorSpeed(turque > 0 ? 30.0f : -30.0f);
        joint->SetMaxMotorTorque(b2Abs(turque));
        // power += b2Abs(turque);
      }
      {
        auto joint = footJoints[j];
        float turque =
            (input[i * 2 + 1] - (angle = joint->GetJointAngle())) * Kp -
            (speed = joint->GetJointSpeed()) * Kd;
        joint->SetMotorSpeed(turque > 0 ? 50.0f : -50.0f);
        joint->SetMaxMotorTorque(b2Abs(turque));
        // power += b2Abs(turque);
      }
    }
  }
  virtual bool update(DataType *output, DataType *reward) override {
    ct = b2Vec2_zero;
    pt = b2Vec2_zero;
    float32 mass = 0;
    auto root = main->GetPosition();
    int index = 0;
    for (int i = 0; i < BODY_COUNT; ++i) {

      if (i > 0) {
        output[index++] = (bodies[i]->GetWorldCenter().x - root.x) / 5;
        output[index++] = (bodies[i]->GetWorldCenter().y - root.y) / 5;
      } else {
        output[index++] = (root.y - position.y) / 5;
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
    for (size_t i = 0; i < 2; i++) {
      footDown[i] *= 0.925f; /*
       output[5 + i * 12] = 0;
       output[6 + i * 12] = 0;*/
      if (testTouch && step > 0)
        for (b2ContactEdge *e = foots[i]->GetContactList(); e; e = e->next) {
          if (e->contact && e->contact->IsTouching()) {
            footDown[i] = 1.0f; /*
             auto m = e->contact->GetManifold();
             for (size_t j = 0; j < m->pointCount; ++j) {
               output[5 + i * 12] += m->points[j].tangentImpulse;
               output[6 + i * 12] += m->points[j].normalImpulse;
               Impulses[i] += m->points[j].normalImpulse * 0.25f;
             }*/
          }
        }
      /*Impulse += b2Abs(output[5 + i * 12]);
      Impulses[i] = b2Abs(output[6 + i * 12]);*/
      output[index] = footDown[i];
      ++index;
    }
    output[index] = (id / 4.0 + 1.0) * 0.25;
    bool mainDown = false;
    if (testTouch && step > 0)
      for (b2ContactEdge *e = main->GetContactList(); e; e = e->next) {
        if (e->contact && e->contact->IsTouching()) {
          mainDown = true;
        }
      }
    DataType add = 0.0;
    int legID = plan[0] > plan[1] ? 0 : 1;
    float forward = foots[1 - legID]->GetWorldPoint(b2Vec2(0, -1.5f)).x -
                    foots[legID]->GetWorldPoint(b2Vec2(0, -1.5f)).x;
    if (forward > 0.2f) {
      if (footDown[1 - legID] == 1.0f) {
        legID = 1 - legID;
        // add += exp(-powf(Impulses[legID] * 0.25f, 2));
        add += exp(-powf(forward - 1.5f, 2)) * 0.75;
        plan[legID] = 1.0f;
        plan[1 - legID] = 0.0f;
      }
    } else if (footDown[1 - legID] == 0.0f) {
      add += 0.2f;
    }
    if (mirror && legID == 1) {
      for (int i = 0; i < STATE_SIZE; ++i) {
        int j = target[i];
        if (j != i) {
          float tmp = output[i];
          output[i] = output[j];
          output[j] = tmp;
        }
      }
    }
    {
      auto v =
          foots[1 - legID]->GetLinearVelocityFromLocalPoint(b2Vec2(0, -1.5));
      add += -powf(fmaxf(v.y, 0) * 0.002, 2.0f) * 0.75;
      add += -powf((5.0f * output[index] - v.x) * 0.0002, 2.0f) * 0.25;
    }
    {
      auto v = foots[legID]->GetLinearVelocityFromLocalPoint(b2Vec2(0, -1.5));
      add += -powf(v.y * 0.03, 2.0f) * 0.75;
      add += -powf(v.x * 0.01, 2.0f) * 0.75;
    }
    // add += -powf(Impulse * 0.2f, 2) * 0.25;
    add += plan[0] * (footDown[0] - 0.5) * 1.5;
    add += plan[1] * (footDown[1] - 0.5) * 1.5;
    // output[index++] = plan[0];
    // output[index++] = plan[1];
    /*if (footDown[0] < 0.8f && footDown[1] < 0.8f) {
            add -= 1.6f;
    }*/
    if (reward) {
      *reward = -powf(0.25 * (ct.x - 4.0f * output[index]), 2) * 0.25f -
                powf(root.y, 2) - powf(ct.y * 0.5f, 2) * 0.2f -
                3 * powf(main->GetAngle(), 2);
      *reward += add;
      if (mainDown) {
        *reward -= 1.5f;
        return true;
      }
    }
    if (pt.x > position.x + 20.0f) {
      for (int i = 0; i < BODY_COUNT; ++i) {
        auto p = bodies[i]->GetPosition();
        p.x -= 20;
        auto r = bodies[i]->GetAngle();
        bodies[i]->SetTransform(p, r);
      }
    }
    return false;
  }
  virtual void display() override {
    g_debugDraw.DrawSegment(b2Vec2(pt.x, 0), b2Vec2(pt.x, id / -4.0 - 1.0),
                            b2Color(1, 1, 1));
    g_debugDraw.DrawSegment(b2Vec2(pt.x + 0.1, 0), b2Vec2(pt.x + 0.1, -ct.x),
                            b2Color(1, 0, 0));
  }
  virtual void destroy() override {}
};
std::vector<int> BallAgent::target;

Agent *createAgent(int id, b2World *w, const b2Vec2 &position,
                   const float32 &angle) {
  BallAgent *agent = new BallAgent();
  agent->create(id, w, position, angle);
  return agent;
}