#include <env/BoxEnv.hpp>
#include <mutex>
float valueAngle = 0;
float jointAngle = 0;
namespace testBox {
void Robot::Init(b2Body *b, const char *model) {
  body = b;
  body->SetAngularDamping(0.0);
  b2MassData data;
  body->GetMassData(&data);
  data.center = b2Vec2_zero;
  body->SetMassData(&data);
  iter = 0;
  train = true;
  maxStep = 32;
  net.initTrainer(model);
  if (model)
    this->model = model;
  else
    this->model = "data/tmp";
  target.SetZero();
  act.resize(net.outputSize, 0.0f);
  input.resize(net.inputSize, 0.0f);
  states.resize(6, 0.0f);
  position = body->GetPosition();
  /*inputs.resize(32 * 32 * net.inputSize, 0.0f);
  for (int y = 0; y < 32; ++y) {
    for (int x = 0; x < 32; ++x) {
      inputs[(y * 32 + x) * net.inputSize + 0] = (x - 15.5) / 15.5;
      inputs[(y * 32 + x) * net.inputSize + 1] = (15.5 - y) / 15.5;
      inputs[(y * 32 + x) * net.inputSize + 2] = 0;
      inputs[(y * 32 + x) * net.inputSize + 3] = 0;
    }
  }*/
  Reset();
  Action();
}
int Robot::Step() {
  int ret = 0;
  if (vStep >= 4) {
    vStep = 0;
    ++step;
    if (step > maxStep) {
      if (!done) {
        reward -= 1.0;
        done = true;
      }
    }
    Update();
    if (train) {
      ret = net.postUpdate(reward, old, real, act, done, states);
    }
    if (done) {
      if (isnanf(act[0]) || isnanf(act[1])) {
        // net.net.check();
        train = false;
      }
      ++iter;
      Reset();
    }
    Action();
  } else
    ++vStep;
  return ret;
}
extern "C" float getReward(const float *target, const float *at, float *state,
                           float *_state) {
  b2Vec2 point(target[3], target[4]);
  b2Transform ot(b2Vec2(at[0], at[1]), b2Rot(at[2]));
  b2Transform nt(b2Vec2(at[3], at[4]), b2Rot(at[5]));
  b2Vec2 d = b2MulT(ot, point);
  state[0] = d.x;
  state[1] = d.y;
  b2Vec2 e = b2MulT(nt, point);
  _state[0] = e.x;
  _state[1] = e.y;
  return d.Length() - e.Length() + (target == at ? 1.0f : 0.0f);
}

void Robot::Update() {
  auto d = body->GetLocalPoint(target);
  d.y -= 2.0;
  auto v = body->GetLocalVector(body->GetLinearVelocity());
  float32 a = body->GetAngularVelocity();
  input[0] = d.x;
  input[1] = d.y;
  input[2] = v.y > 0.0f ? 1.0f : 0.0f; /*
         input[3] = v.y*0.1;
         input[4] = a*0.1;*/
  real = input;

  {
    auto d = body->GetWorldPoint(b2Vec2(0.0f, 2.0f));
    states[3] = d.x;
    states[4] = d.y;
    states[5] = body->GetAngle();
  }
  if (filp)
    input[0] = -input[0];
  start = shape;
  shape = -sqrt(d.x * d.x + d.y * d.y /*+
            a   * a * 0.3 + v.x * v.x*0.2 + v.y * v.y * 0.2*/);
  reward = shape - start;
  if (shape > -0.25) {
    done = true;
    reward += 1;
  }
}
void Robot::Reset() {
  float32 value = drand48();
  step = 0;
  vStep = 0;
  keepTime = 0;
  // target = b2Rot(drand48() * b2_pi * 2).GetXAxis();
  target.Set((value - 0.5) * 18 + position.x, position.y);

  body->SetTransform(position, b2_pi * drand48() * 2);
  body->SetLinearVelocity(b2Vec2_zero);
  body->SetAngularVelocity(0);
  Update();
  done = false;
  filp = !filp && train;
  start = shape;
  // shape = 0.0f;
}
void Robot::Action() {
  old = real;
  {
    states[0] = states[3];
    states[1] = states[4];
    states[2] = states[5];
  }
  act = net.preUpdate(input);
  if (filp)
    act[0] = -act[0];
  if (body) {
    auto speed = act[0] = fminf(fmaxf(act[0], -2.0f), 2.0f);
    auto force = act[1] = fminf(fmaxf(act[1], -0.5f), 5.0f);
    const auto &wv = body->GetWorldVector(b2Vec2(0, force));
    /*body->ApplyForceToCenter(wv, false);
    body->ApplyTorque(speed, true);*/
    body->SetAngularVelocity(speed * force * 0.5f);
    body->SetLinearVelocity(wv);
    // body->SetTransform(body->GetPosition()+b2Vec2(force,0), 0);
  }
}
void Robot::SaveNet() { net.save(model.c_str()); }
static Robot robot_;
static b2World *w = nullptr;
static b2Body *m_groundBody;
static bool running = false;
static float time_step = 0.0333f;
static void box2d_init() {
  if (w)
    return;
  b2Vec2 g(0, -10);
  w = new b2World(g);
  b2BodyDef def;
  def.type = b2_staticBody;
  m_groundBody = w->CreateBody(&def);
  b2PolygonShape box; // = new b2PolygonShape();
  box.SetAsBox(10.f, .2f, b2Vec2(0, -.2), 0);
  m_groundBody->CreateFixture(&box, 1);
  {
    def.allowSleep = false;
    def.angle = 0.0;
    b2Filter filter;
    filter.groupIndex = -1;
    // init_pd(&body);
    {
      def.type = b2_dynamicBody;
      box.SetAsBox(0.5f, 1.5f);
      def.gravityScale = 0.0;
      def.position.Set(0.0, 12.0);
      auto a = w->CreateBody(&def);
      a->CreateFixture(&box, 1)->SetFilterData(filter);
      robot_.Init(a);
      /*if (1) {
        b2RevoluteJointDef j;
        j.Initialize(m_groundBody, a, b2Vec2(def.position.x, def.position.y));
        auto b = (b2RevoluteJoint *)w->CreateJoint(&j);
      }*/
    }
  }
}
static void box2d_quit() {
  if (w)
    delete w;
  w = nullptr;
}
static void sync_Network();
static Robot *target = 0;
static int outputIter = 0;
static int runStep = 0;
static void *box2d_step(void *rb) {
  box2d_init();
  target = (Robot *)rb;
  if (target) {
    outputIter = 1;
    robot_.net.s = target->net.s;
  }
  running = true;
  while (running) {
    w->Step(time_step, 8, 5);
    if (robot_.Step()) {
      ++runStep;
      if (outputIter && runStep >= outputIter) {
        runStep = 0;
        sync_Network();
      }
    }
  }
  box2d_quit();
  // robot.SaveNet();
  sync_Network();
  runStep = 0;
  target = 0;
  outputIter = 0;
  return 0;
}
static std::mutex lock;
static pthread_t pt;
void setRunning(bool b) {
  lock.lock();
  running = b;
  lock.unlock();
}
void syncNetwork(int saveIter) {
  lock.lock();
  if (running) {
    if (saveIter) {
      outputIter = saveIter;
    } else {
      outputIter = 0;
      if (target) {
        target->net.exp = robot_.net.exp;
        target->net.tuple->returns = robot_.net.tuple->returns;
        target->net.tuple->obs = robot_.net.tuple->obs;
        target->net.tuple->adv = robot_.net.tuple->adv;
        target->net.tuple->rewards = robot_.net.tuple->rewards;
        target->net.tuple->position = robot_.net.tuple->position;
        Net::CopyModel(*robot_.net.value.tlocal, *target->net.value.tlocal);
        target->net.value.syncNet();
        Net::CopyModel(*robot_.net.net.tlocal, *target->net.net.tlocal);
        target->net.net.syncNet();
      }
    }
  }
  lock.unlock();
}
static void sync_Network() {
  if (target) {
    lock.lock();
    if (!running) {
      target->net.s = robot_.net.s;
      target->train = true;
      target->maxStep = robot_.maxStep;
    }
    target->net.exp = robot_.net.exp;
    target->net.tuple->adv = robot_.net.tuple->adv;
    target->net.tuple->returns = robot_.net.tuple->returns;
    target->net.tuple->obs = robot_.net.tuple->obs;
    target->net.tuple->rewards = robot_.net.tuple->rewards;
    target->net.tuple->position = robot_.net.tuple->position;
    Net::CopyModel(*robot_.net.value.tlocal, *target->net.value.tlocal);
    target->net.value.syncNet();
    Net::CopyModel(*robot_.net.net.tlocal, *target->net.net.tlocal);
    target->net.net.syncNet();
    lock.unlock();
  }
}
void saveNetwork() {
  lock.lock();
  if (running) {
    robot_.SaveNet();
  }
  lock.unlock();
}
int runLearnning(Robot *from) {
  lock.lock();
  if (running) {
    lock.unlock();
    if (running)
      return 2;
  } else
    lock.unlock();
  return pthread_create(&pt, 0, &box2d_step, from) ? 1 : 0;
}
} // namespace testBox