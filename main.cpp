#include "main.hpp"
#include <conio.h>
#include <env/BoxEnv.hpp>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

tExpTuple::tExpTuple(int size, int state, int action) {
	states.resize(size * state, 0.0f);
	n_states.resize(size * state, 0.0f);
	actions.resize(size * action, 0.0f);
	rewards.resize(size, 0.0f);
	returns.resize(size, 0.0f);
	values.resize(size, 0.0f);
	dones.resize(size, false);
	adv.resize(size, 0.0f);
	maxStep = size;
	position = 0;
	scale = 1;
	for (auto &i : states)
		i = drand48() - 0.5;
}
void Trainer::initTrainer(int s, int a, int type) {
	startupCaffe();
	this->type = type;
	value.makeNet(s, 1, "solver_value.txt");
	inputSize = s;
	outputSize = a;
	scale.resize(a, 0.5f);
	if (type)
		net.makeNet(s, a, "solver_awr.txt");
	else
		net.makeNet(s, a, "solver_ppo.txt");
}
void Trainer::load(const char *model) {
	if (model) {
		net.Load(std::string(model) + std::string("_actor.bin"));
		value.Load(std::string(model) + std::string("_critic.bin"));
	}
	// sprintf(statString, "exp: %f", s);
	/*int comp = 0;
	float loss = 1.0;
	do {
			loss = net.trainNet(input, label, w);
			++comp;
	} while (loss > 0.000002);
	{
			net.syncNet();
			auto v = net.getValue(input);
			for (auto &i : input) printf("%f\t", i);
			printf("\n");
			for (auto &i : v) printf("%f\t", i);
			printf("\nloss= %f, use %d\n", loss, comp);
	}*/
}
const std::vector<float> &Trainer::preUpdate(const std::vector<float> &input) {
	// s = shape;
	// state = input;
	auto &act = net.getValue(input, "std", scale);
	int cnt = scale.size();
	if (act.size() == cnt) {
		for (int j = 0; j < cnt; ++j) {
			auto &i = act[j];
			i = cMathUtil::RandDoubleNorm(i, scale[j]);
		}
	}
	return act;
}
float NormalLogp(float scale, float x) {
	float sx = x / scale;
	return -0.5 * sx * sx - 0.5 * logf(M_PI + M_PI) - logf(scale);
}
float NormalProb(float scale, float x) {
	float sx = x / scale;
	return expf(-0.5 * sx * sx - 0.5 * logf(M_PI + M_PI) - logf(scale));
}
float NormalTD(float scale, float x) {
	return (-sqrtf(0.5 * M_1_PI) * x * expf(-0.5 * x * x)) /
		   (scale * scale * scale);
}
extern "C" float getReward(const float *target, const float *at, float *state,
						   float *_state);
int Trainer::postUpdate(int id, float shape, const std::vector<float> &o_input,
						const std::vector<float> &n_input,
						const std::vector<float> &act, bool done) {
	// auto o = value.getValue(state)[0];
	// exp = cMathUtil::EvalGaussian(0, 1.0, 0.0);
	while (tuples.size() <= id)
		tuples.push_back(std::shared_ptr<tExpTuple>(
			new tExpTuple(256, inputSize, outputSize)));
	auto tuple = tuples[id];
	for (int i = 0; i < inputSize; ++i) {
		tuple->states[i + tuple->position * inputSize] = o_input[i];
		tuple->n_states[i + tuple->position * inputSize] = n_input[i];
	}
	for (int i = 0; i < outputSize; ++i) {
		tuple->actions[i + tuple->position * outputSize] = act[i];
	}
	tuple->rewards[tuple->position] = shape;
	// tuple->values[tuple->position] = o;
	tuple->dones[tuple->position] = done;

	tuple->position = (tuple->position + 1) % tuple->maxStep;
	if (tuple->position == 0) {
		auto &reward = tuple->rewards;
		auto &value = tuple->values;
		// auto &_value = tuple->_values;
		auto &done = tuple->dones;
		auto &gaelam = tuple->adv;
		auto &ret = tuple->returns;
		/*if (drand48() > 0.5f)*/if(0) {
			auto &out = tuple->states;
			auto &output = tuple->n_states;
			int target = -1;
			for (int t, i = tuple->maxStep; i > 0; --i) {
				t = (i + tuple->position - 1) % tuple->maxStep;
				if (done[t]) {
					target = t;
					/*if (target == -1) {
						target = t;
					}  else break;*/
				}
				if (target != -1) {
					double q = 0;
					for (int j = 0; j < 5; ++j) {
						out[t * inputSize + 31 + j] =
						output[t * inputSize + 31 + j] =
							output[target * inputSize + j];
						q += pow(output[t * inputSize + j] -
									 output[target * inputSize + j],
								 2);
					}
					reward[t] = ::exp(-q*3);
				}
				/*getReward(tuple->world.data() + target * ssize,
										tuple->world.data() + t * ssize,
										tuple->states.data() + t * inputSize,
										tuple->_states.data() + t *
				   inputSize);*/
			}
		}
		this->value.getValues(tuple->states, value, 256);
		auto nv = this->value.getValue(n_input)[0];
		const float gamma = 0.975;
		const float lam = 0.935;
		float lastgaelam = 0.0f;
		float total = 0.0;
		float stddev = 0.0;
		for (int t, i = tuple->maxStep; i > 0; --i) {
			t = (i + tuple->position - 1) % tuple->maxStep;
			float nonterminal = done[t] ? 0.0f : 1.0f;
			ret[t] = reward[t] + gamma * nonterminal * (nv + lam * lastgaelam);
			gaelam[t] = lastgaelam = ret[t] - (nv = value[t]);
			total += gaelam[t];
		}
		total /= tuple->maxStep;
		for (auto &gae : gaelam) {
			gae -= total;
			stddev += gae * gae;
		}
		stddev = sqrtf(stddev / tuple->maxStep) + 0.001f;
		for (auto &gae : gaelam) {
			gae /= stddev;
		}
		this->value.LoadTrainData(tuple->states, tuple->returns,
								  std::vector<float>());
		exp = this->value.trainNet(10);
		this->value.getValues(tuple->states, value, 256);
		if (type == 1) {
			nv = this->value.getValue(n_input)[0];
			lastgaelam = 0.0f;
			total = 0.0;

			for (int t, i = tuple->maxStep; i > 0; --i) {
				t = (i + tuple->position - 1) % tuple->maxStep;
				float nonterminal = done[t] ? 0.0f : 1.0f;
				ret[t] =
					reward[t] + gamma * nonterminal * (nv + lam * lastgaelam);
				gaelam[t] = lastgaelam = ret[t] - (nv = value[t]);
				total += gaelam[t];
			}
			if (id == 0) {
				/*float total_ret = 0.0;
				for (auto &r : ret)
				  total_ret += r;
				total_ret /= tuple->maxStep;*/
				stddev = 0.0;
				for (auto &r : ret)
					stddev += powf(r, 2.0);
				stddev = sqrtf(stddev / tuple->maxStep) + 0.001f;
				tuple->scale = 1.0 / stddev;
			}
			total /= tuple->maxStep;
			stddev = 0.0;
			for (auto &gae : gaelam) {
				gae -= total;
				stddev += gae * gae;
			}
			stddev = sqrtf(stddev / tuple->maxStep) + 0.001f;
			for (auto &gae : gaelam) {
				gae /= stddev;
			}
		}
		net.LoadTrainData(tuple->states, tuple->actions, gaelam);
		if (type == 0)
			net.SetTrainParam("logprob", "logoldprob");
		this->net.trainNet(10);
		return 1;
	}
	// s = s - shape;
	return 0;
}
void Trainer::save(const char *model) {
	if (model) {
		net.Save(std::string(model) + std::string("_actor.bin"));
		value.Save(std::string(model) + std::string("_critic.bin"));
	}
}
void Trainer::shutDown() { shutDownCaffe(); }