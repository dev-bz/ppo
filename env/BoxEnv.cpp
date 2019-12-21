#include <env/BallAgent.hpp>
#include <env/BoxEnv.hpp>
#include <mutex>
float g_time = 0;

namespace testBox {
// static const std::vector<DataType> empty;

std::vector<DataType> n_state, m_new, m_old, m_action;
void Robot::Init(b2World *m_world, int pgType, const char *model) {
  int STATE_SIZE = agentStateSize();
  int ACTION_SIZE = agentActionSize();
  n_state.resize(STATE_SIZE);
  m_new.resize(STATE_SIZE);
  m_old.resize(STATE_SIZE);
  m_action.resize(ACTION_SIZE);
  net.initTrainer(STATE_SIZE, ACTION_SIZE, pgType);
  if (model)
    this->model = model;
  else
    this->model = "data/tmp";
  net.load(this->model.c_str());
  b2BodyDef def;
  b2EdgeShape ground;
  ground.Set(b2Vec2(-100, 0), b2Vec2(300, 0));
  int rot = int(pow(AGENT_COUNT, 0.5));
  int i = 0;
  m_world->CreateBody(&def)->CreateFixture(&ground, 0.0f)->SetFriction(10.0f);
  obs.resize(AGENT_COUNT);
  action.resize(AGENT_COUNT);
  for (int i = 0; i < AGENT_COUNT; ++i) {
    agents[i] = createAgent(i, m_world, b2Vec2(0, 2), 0.0f);
    obs[i].resize(STATE_SIZE);
    action[i].resize(ACTION_SIZE);
    agents[i]->reset(obs[i].data());
  }
  /*  int count = 0;
    while (count < AGENT_COUNT) {
    def.position.y = i * 5;
    m_world->CreateBody(&def)->CreateFixture(&ground, 0.0f)->SetFriction(10.0f);
    for (int j = 0; j < rot; ++j) {
      if (count < AGENT_COUNT) {
        agents[count] =
            createAgent(count, m_world, b2Vec2(j * 5, 2 + i * 5), 0.0f);
        obs[count].resize(STATE_SIZE);
        action[count].resize(ACTION_SIZE);
        agents[count]->reset(obs[count].data());
        ++count;
      }
    }
    ++i;
  }
  def.position.y = i * 5;
  m_world->CreateBody(&def)->CreateFixture(&ground, 0.0f)->SetFriction(10.0f);*/
  train = 1;
  trains = 0;
  code = 0;
}
void Robot::Quit() { net.shutDown(); }
int Robot::Step() {
  g_time += 0.033f;
  int ret = 0;
  for (int i = 0; i < AGENT_COUNT; ++i) {
    DataType reward = 0;
    bool done;
    if (train) {
      done = agents[i]->update(n_state.data(), &reward);
      if (mirrorAgent(m_old.data(), m_new.data(), m_action.data(), obs[i].data(), n_state.data(), action[i].data()))
        ret |= net.postUpdate(i + AGENT_COUNT, reward, m_old, m_new, m_action, done);
      ret |= net.postUpdate(i, reward, obs[i], n_state, action[i], done);
    } else
      done = agents[i]->update(n_state.data(), nullptr);
    if (done)
      agents[i]->reset(obs[i].data());
    else
      obs[i] = n_state;
  }
  trains += ret;
  return ret;
}
void Robot::Action() {
  for (int i = 0; i < AGENT_COUNT; ++i) {
    action[i] = net.preUpdate(obs[i]);
    agents[i]->apply(action[i].data());
  }
}
void Robot::SaveNet() { net.save(model.c_str()); }
static Robot robot_;
static b2World *w = nullptr;
static bool running = false;
static float time_step = 0.016667f;
static void box2d_init() {
  if (w)
    return;
  b2Vec2 g(0, -10);
  w = new b2World(g);
  threadAt(1);
  robot_.Init(w, 1, "data/sync");
  threadAt(0);
}
static void box2d_quit() {
  robot_.Quit();
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
    syncNetwork(-1);
    outputIter = 3;
  }
  running = true;
  while (running) {
    robot_.Action();
    w->Step(time_step, 25, 10);
    if (robot_.Step()) {
      ++runStep;
      if (outputIter && runStep >= outputIter) {
        runStep = 0;
        sync_Network();
      }
    }
  }
  // robot.SaveNet();
  sync_Network();
  box2d_quit();
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
  if (saveIter < 0) {
    if (target) {
      robot_.trains = target->trains;
      target->code = Net::CopyModel(*target->net.value.tlocal, *robot_.net.value.tlocal);
      target->code |= Net::CopyModel(*target->net.net.tlocal, *robot_.net.net.tlocal); /*
                   target->code =
                       Net::CopyModel(*target->net.value.batch, *robot_.net.value.batch);
                   target->code |=
                       Net::CopyModel(*target->net.net.batch, *robot_.net.net.batch);
                   target->code =
                       Net::CopyModel(*target->net.value.local, *robot_.net.value.local);
                   target->code |=
                       Net::CopyModel(*target->net.net.local, *robot_.net.net.local);*/
      for (auto t : robot_.net.tuples)
        t->position = 0;
      target->code |= 4;
    }
  } else if (running) {
    if (saveIter) {
      outputIter = saveIter;
    } else {
      outputIter = 0;
      if (target) {
        target->net.exp = robot_.net.exp;
        target->trains = robot_.trains;
        if (target->net.tuples.size() > 0 && robot_.net.tuples.size() > 0) {
          target->net.tuples[0]->returns = robot_.net.tuples[0]->returns;
          target->net.tuples[0]->values = robot_.net.tuples[0]->values;
          target->net.tuples[0]->adv = robot_.net.tuples[0]->adv;
          target->net.tuples[0]->rewards = robot_.net.tuples[0]->rewards;
          target->net.tuples[0]->dones = robot_.net.tuples[0]->dones;
          target->net.tuples[0]->scale = robot_.net.tuples[0]->scale;
        }
        target->code = Net::CopyModel(*robot_.net.value.tlocal, *target->net.value.tlocal);
        target->code |= Net::CopyModel(*robot_.net.net.tlocal, *target->net.net.tlocal);
      }
    }
  }
  lock.unlock();
}
static void sync_Network() {
  if (target) {
    lock.lock();
    if (!running) {
      for (auto t : target->net.tuples)
        t->position = 0;
      target->train = 1;
    }
    target->net.exp = robot_.net.exp;
    target->trains = robot_.trains;
    if (target->net.tuples.size() > 0 && robot_.net.tuples.size() > 0) {
      target->net.tuples[0]->returns = robot_.net.tuples[0]->returns;
      target->net.tuples[0]->values = robot_.net.tuples[0]->values;
      target->net.tuples[0]->adv = robot_.net.tuples[0]->adv;
      target->net.tuples[0]->rewards = robot_.net.tuples[0]->rewards;
      target->net.tuples[0]->dones = robot_.net.tuples[0]->dones;
      target->net.tuples[0]->scale = robot_.net.tuples[0]->scale;
    }
    target->code = Net::CopyModel(*robot_.net.value.tlocal, *target->net.value.tlocal);
    target->code |= Net::CopyModel(*robot_.net.net.tlocal, *target->net.net.tlocal);
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