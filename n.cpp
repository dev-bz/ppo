#include "n.hpp"
#include <android/log.h>
#include <caffe/caffe.hpp>
#include <caffe/net.hpp>
#include <caffe/sgd_solvers.hpp>
#define LOGI(...)                                                              \
	((void)__android_log_print(ANDROID_LOG_INFO, "native-activity",            \
							   __VA_ARGS__))
#define LOGW(...)                                                              \
	((void)__android_log_print(ANDROID_LOG_WARN, "native-activity",            \
							   __VA_ARGS__))

void InitCaffe(char **argv);
int Net::GetBatchSize() {
	int batch_size = 0;
	{
		const auto &input_blobs = tlocal->input_blobs();
		const auto &input_blob = input_blobs[0];
		batch_size = input_blob->shape(0);
	}
	return batch_size;
}
void Net::SetTrainParam(const char *from, const char *to) {
	auto &layers = tlocal->layer_names();
	int end = 0;
	for (auto &l : layers) {
		if (l == from)
			break;
		++end;
	}
	if (end == layers.size())
		return;
	tlocal->ForwardTo(end);
	auto &src_blob = tlocal->blob_by_name(from);
	auto &dst_blob = tlocal->blob_by_name(to);
	end = src_blob->count();
	if (dst_blob->count() == end) {
		dst_blob->CopyFrom(*src_blob);
	}
}
void Net::SetTrainParam(const std::vector<float> &param) {
	int blobs = tlocal->input_blobs().size();
	if (blobs > 3) {
		auto dt = param.data();
		int jc = param.size();
		auto &i = tlocal->input_blobs()[3];
		int ct = i->count();
		float *td = i->mutable_cpu_data();
		for (int j = 0; j < ct; ++j) {
			td[j] = dt[j % jc];
		}
	}
}
float Net::GetTrainResult() {
	int blobs = tlocal->input_blobs().size();
	if (blobs > 3) {
		auto &i = tlocal->input_blobs()[3];
		if (i->count() > 4)
			return i->cpu_data()[4];
	}
	return 0;
}
void Net::LoadTrainData(const std::vector<float> &X,
						const std::vector<float> &Y,
						const std::vector<float> &W) {
	int blobs = tlocal->input_blobs().size();
	{
		auto dt = X.data();
		int jc = X.size();
		auto &i = tlocal->input_blobs()[0];
		int ct = i->count();
		float *td = i->mutable_cpu_data();
		for (int j = 0; j < ct; ++j) {
			td[j] = dt[j % jc];
		}
	}
	if (blobs > 1) {
		auto dt = Y.data();
		int jc = Y.size();
		auto &i = tlocal->input_blobs()[1];
		int ct = i->count();
		float *td = i->mutable_cpu_data();
		for (int j = 0; j < ct; ++j) {
			td[j] = dt[j % jc];
		}
	}
	int jc = W.size();
	if (blobs > 2 && jc > 0) {
		auto dt = W.data();
		auto &i = tlocal->input_blobs()[2];
		int ct = i->count();
		float *td = i->mutable_cpu_data();
		for (int j = 0; j < ct; ++j) {
			td[j] = dt[j % jc];
		}
	}
}
int Net::CopyParams(const std::vector<caffe::Blob<float> *> &src_params,
					const std::vector<caffe::Blob<float> *> &dst_params) {
	int num_blobs = static_cast<int>(src_params.size());
	int dst_blob_count_blobs = static_cast<int>(dst_params.size());
	if (num_blobs != dst_blob_count_blobs)
		return 1;
	int bad = 0;
	for (int b = 0; b < num_blobs; ++b) {
		auto src_blob = src_params[b];
		auto dst_blob = dst_params[b];
		int src_blob_count = src_blob->count();
		int dst_blob_count = dst_blob->count();
		if (src_blob_count != dst_blob_count) {
			LOGI("[%s %s]", src_blob->shape_string().c_str(),
				 dst_blob->shape_string().c_str());
			bad = 2;
		}
	}
	if (bad == 0)
		for (int b = 0; b < num_blobs; ++b) {
			auto src_blob = src_params[b];
			auto dst_blob = dst_params[b];
			dst_blob->CopyFrom(*src_blob);
		}
	return bad;
}
int Net::CopyModel(const caffe::Net<float> &src, caffe::Net<float> &dst) {
	const auto &src_params = src.learnable_params();
	const auto &dst_params = dst.learnable_params();
	return CopyParams(src_params, dst_params);
}
void Net::syncNet(bool train) {
	if (tlocal != nullptr) {
		if (train) {
			CopyModel(*tlocal.get(), *local.get());
			CopyModel(*tlocal.get(), *batch.get());
		} else {
			CopyModel(*local.get(), *tlocal.get());
			CopyModel(*local.get(), *batch.get());
		}
	}
}
static char proc[64] = {"a.out"};
static int init = 0;
void startupCaffe() {
	if (init == 0) {
		char *argv[1] = {proc};
		// InitCaffe(argv);
	}
	++init;
}
void shutDownCaffe() {
	// if (init == 1) google::ShutdownGoogleLogging();
	if (init > 0)
		--init;
}
void resizeNetParam(caffe::NetParameter *n, int i, int o) {
	for (int id = 0; id < n->layer_size(); ++id) {
		auto layer = n->mutable_layer(id);
		if (layer->name() == "input") {
			if (layer->has_input_param()) {
				auto m = layer->mutable_input_param();
				if (i > 0 && m->shape_size() > 0)
					if (i != m->shape(0).dim(1))
						m->mutable_shape(0)->set_dim(1, i);
				if (o > 0 && m->shape_size() > 1)
					if (o != m->shape(1).dim(1))
						m->mutable_shape(1)->set_dim(1, o);
			}
		} else if (o > 0) {
			if (layer->name() == "output") {
				if (layer->has_inner_product_param())
					if (o != layer->inner_product_param().num_output())
						layer->mutable_inner_product_param()->set_num_output(o);
			} else if (layer->name() == "logstd") {
				auto m = layer->mutable_parameter_param();
				if (m->has_shape()) {
					if (o != m->shape().dim(1))
						m->mutable_shape()->set_dim(1, o);
				}
			}
		}
	}
}
void Net::makeNet(int i, int o, const char *file, bool solver) {
	const char *SolverType;
	caffe::NetParameter net_param;
	caffe::SolverParameter param;
	if (solver) {
		caffe::ReadProtoFromTextFileOrDie(file, &param);
		file = nullptr;
		{
			auto n = param.mutable_net_param();
			if (param.has_net())
				caffe::ReadProtoFromTextFileOrDie(param.net(), n);
			resizeNetParam(n, i, o);
			net_param.CopyFrom(*n);
			param.clear_net();
		}
		caffe::SolverParameter_SolverType type = param.solver_type();
		SolverType = param.type().c_str();
		if (strcmp("SGD", SolverType) == 0) {
			type = caffe::SolverParameter_SolverType_SGD;
		} else if (strcmp("Nesterov", SolverType) == 0) {
			type = caffe::SolverParameter_SolverType_NESTEROV;
		} else if (strcmp("AdaGrad", SolverType) == 0) {
			type = caffe::SolverParameter_SolverType_ADAGRAD;
		} else if (strcmp("RMSProp", SolverType) == 0) {
			type = caffe::SolverParameter_SolverType_RMSPROP;
		} else if (strcmp("AdaDelta", SolverType) == 0) {
			type = caffe::SolverParameter_SolverType_ADADELTA;
		} else if (strcmp("Adam", SolverType) == 0) {
			type = caffe::SolverParameter_SolverType_ADAM;
		}
		switch (type) {
		case caffe::SolverParameter_SolverType_SGD:
			slocal.reset(new caffe::SGDSolver<float>(param));
			break;
		case caffe::SolverParameter_SolverType_NESTEROV:
			slocal.reset(new caffe::NesterovSolver<float>(param));
			break;
		case caffe::SolverParameter_SolverType_ADAGRAD:
			slocal.reset(new caffe::AdaGradSolver<float>(param));
			break;
		case caffe::SolverParameter_SolverType_RMSPROP:
			slocal.reset(new caffe::RMSPropSolver<float>(param));
			break;
		case caffe::SolverParameter_SolverType_ADADELTA:
			slocal.reset(new caffe::AdaDeltaSolver<float>(param));
			break;
		case caffe::SolverParameter_SolverType_ADAM:
			slocal.reset(new caffe::AdamSolver<float>(param));
			break;
		default:
			LOG(FATAL) << "Unknown SolverType: " << type;
		}
		tlocal = slocal->net();
		int blobs = tlocal->input_blobs().size();
		if (blobs > 2) {
			auto &i = tlocal->input_blobs()[2];
			int ct = i->count();
			float *td = i->mutable_cpu_data();
			for (int j = 0; j < ct; ++j)
				td[j] = 1.0;
		}
		// slocal->Restore("ppo_iter_190.solverstate");
		SolverType = slocal->type();
		if (slocal->test_nets().size() > 0) {
			local = slocal->test_nets()[0];
		}
		if (slocal->test_nets().size() > 1) {
			batch = slocal->test_nets()[1];
		}
	} else {
		caffe::ReadProtoFromTextFileOrDie(file, &net_param);
		// resizeNetParam(&net_param, i, o);
		file = nullptr;
	}
	net_param.mutable_state()->set_phase(caffe::TEST);
	if (local == nullptr) {
		local.reset(new caffe::Net<float>(net_param));
		local->ShareTrainedLayersWith(tlocal.get());
	}
	if (batch == nullptr) {
		batch.reset(new caffe::Net<float>(net_param));
		auto &blob = batch->blob_by_name("input");
		auto shape = blob->shape();
		shape[0] = 256;
		blob->Reshape(shape);
		batch->ShareTrainedLayersWith(tlocal.get());
	}
	/*i = local->blob_by_name("input")->count();
	o = local->blob_by_name("output")->count();*/
}

std::vector<float> &Net::getValue(const std::vector<float> &input,
								  const char *second,
								  std::vector<float> &output) {
	auto &b = local->input_blobs()[0];
	auto dt = input.data();
	int ix = input.size();
	int ct = b->count();
	if (ct > ix)
		ct = ix;
	float *td = b->mutable_cpu_data();
	for (int j = 0; j < ct; ++j) {
		td[j] = dt[j];
	}

	float loss = 0;
	local->Forward(&loss);
	{
		auto &i = local->output_blobs()[0];
		auto cnt = i->count();
		if (local_data.size() != cnt)
			local_data.resize(cnt);
		auto d = i->cpu_data();
		for (int j = 0; j < cnt; ++j) {
			local_data[j] = d[j];
		}
	}
	{
		auto &i = local->output_blobs()[1];
		auto cnt = i->count();
		if (output.size() != cnt)
			output.resize(cnt);
		auto d = i->cpu_data();
		for (int j = 0; j < cnt; ++j) {
			output[j] = d[j];
		}
	}
	return local_data;
}
std::vector<float> &Net::getValue(const std::vector<float> &input) {
	auto &b = local->input_blobs();
	// for (int v = 0; v < ct; ++v)
	{
		auto dt = input.data();
		int ix = 0;
		for (auto &i : b) {
			int ct = i->count();
			float *td = i->mutable_cpu_data();
			for (int j = 0; j < ct; ++j) {
				td[j] = dt[ix++];
			}
		}
		float loss = 0;
		local->Forward(&loss);
		auto &i = local->output_blobs()[0];
		auto cnt = i->count();
		if (local_data.size() != cnt)
			local_data.resize(ix);
		auto d = i->cpu_data();
		for (int j = 0; j < cnt; ++j) {
			local_data[j] = d[j];
		}
	}
	return local_data;
}
float Net::getValues(const std::vector<float> &input,
					 std::vector<float> &output, int batchSize) {

	{
		const auto &input_blobs = batch->input_blobs();
		int batch_size = input_blobs[0]->shape(0);
		if (batch_size != batchSize) {
			for (auto blob : input_blobs) {
				auto shape = blob->shape();
				shape[0] = batchSize;
				blob->Reshape(shape);
			}
			batch->ShareTrainedLayersWith(tlocal.get());
		}
	}
	auto &b = batch->input_blobs();
	// for (int v = 0; v < ct; ++v)
	float loss = 0;
	if (input.size() > 0) {
		auto dt = input.data();
		int ix = 0;
		for (auto &i : b) {
			int ct = i->count();
			float *td = i->mutable_cpu_data();
			for (int j = 0; j < ct; ++j) {
				td[j] = dt[ix++];
			}
		}
		batch->Forward(&loss);
		auto &o = batch->output_blobs();
		ix = 0;
		for (auto &i : o) {
			auto cnt = i->count();
			if (output.size() != cnt)
				output.resize(cnt);
			auto d = i->cpu_data();
			for (int j = 0; j < cnt; ++j, ++ix) {
				output[ix] = d[j];
			}
		}
	}
	return loss;
}
float Net::trainNet(int iters) {
	// LoadTrainData(input, target, w);
	float loss = 0;
	// for (int e = 0; e < 100; ++e) {
	slocal->Step(iters);
	// CopyModel(*tlocal.get(), *local.get());
	auto &o = tlocal->output_blobs();
	int cnt = 0;
	for (auto &i : o) {
		auto d = i->cpu_data();
		for (int j = 0; j < i->count(); ++j) {
			loss += d[j];
		}
		cnt += i->count();
	}
	loss /= cnt;
	// if (loss < 0.05) break;
	//}
	return loss;
}
void Net::Load(const std::string &model_file) {
	if (model_file != "") {
		if (local != nullptr) {
			if (model_file.size() >= 4 &&
				model_file.compare(model_file.size() - 4, 4, ".bin") == 0) {
				FILE *f = fopen(model_file.c_str(), "rb");
				if (f) {
					for (auto &layer : local->layers()) {
						auto &b = layer->blobs();
						if (b.size() > 0) {
							// const caffe::LayerParameter &ppp =
							// layer->layer_param();
							for (auto &blob : b) {
								auto count = blob->count();
								auto data = blob->mutable_cpu_data();
								fread(data, sizeof(float), count, f);
							}
						}
					}
					fclose(f);
				}
			}
			//** mNet->CopyTrainedLayersFromHDF5(model_file);
			// LoadScale(GetOffsetScaleFile(model_file));
			syncNet(false);
		} else {
			printf("Net structure has not been initialized\n");
			assert(false);
		}
	}
}
void Net::Save(const std::string &model_file) {
	if (model_file != "") {
		if (local != nullptr) {
			syncNet(true);
			if (model_file.size() >= 4 &&
				model_file.compare(model_file.size() - 4, 4, ".bin") == 0) {
				FILE *f = fopen(model_file.c_str(), "wb");
				if (f) {
					for (auto &layer : local->layers()) {
						auto &b = layer->blobs();
						if (b.size() > 0) {
							// const caffe::LayerParameter &ppp =
							// layer->layer_param();
							for (auto &blob : b) {
								auto count = blob->count();
								auto data = blob->cpu_data();
								fwrite(data, sizeof(float), count, f);
							}
						}
					}
					fclose(f);
				}
			}
			//** mNet->CopyTrainedLayersFromHDF5(model_file);
			// LoadScale(GetOffsetScaleFile(model_file));
		} else {
			printf("Net structure has not been initialized\n");
			assert(false);
		}
	}
}
float Net::check(float weight_decay) {
	if (tlocal != nullptr) {
		float max = 0;
		int nan_count = 0;
		int nan_diff_count = 0;

		int nan_all_count = 0;
		float max_diff = 0;
		/*for (auto &layer : tlocal->layers())*/ {
			auto &b = tlocal->blobs();
			for (auto &blob : b) {
				auto count = blob->count();
				nan_all_count += count;
				{
					auto data = blob->mutable_cpu_data();
					for (int i = 0; i < count; ++i) {
						max = fmaxf(fabsf(data[i]), max);
						if (isnanf(data[i]) || isinf(data[i])) {
							data[i] = drand48() * 0.001 - 0.0005;
							++nan_count;
						}
						// data[i] *= weight_decay;
					}
					// data[rand() % count] = drand48() * 0.001;
				}
				{
					auto data = blob->cpu_diff();
					for (int i = 0; i < count; ++i) {
						max_diff = fmaxf(fabsf(data[i]), max_diff);
						if (isnanf(data[i]) || isinf(data[i])) {
							++nan_diff_count;
						}
					}
				}
			}
		}
		// printf("check: %f, %f (%d, %d)/%d\n", max, max_diff, nan_count,
		// nan_diff_count, nan_all_count);
		return max_diff;
	}
	//** mNet->CopyTrainedLayersFromHDF5(model_file);
	// LoadScale(GetOffsetScaleFile(model_file));
	else {
		printf("Net structure has not been initialized\n");
		assert(false);
	}
}
void InitCaffe(char **argv) {
	if (argv != NULL) {
		FLAGS_log_dir = "log";
		FLAGS_v = 3;
		// FLAGS_logfile_mode = 0;
		// FLAGS_alsologtostderr = 0;
		int caffe_argc = 1; // hack
		caffe::GlobalInit(&caffe_argc, &argv);
	}
}
extern "C" void testPPO() {
	const char *solver = "solver_t4value.txt";
	std::shared_ptr<caffe::Solver<float>> slocal;
	const char *SolverType;
	if (solver) {
		caffe::SolverParameter param;
		caffe::ReadProtoFromTextFileOrDie(solver, &param);
		caffe::SolverParameter_SolverType type = param.solver_type();
		SolverType = param.type().c_str();
		if (strcmp("SGD", SolverType) == 0) {
			type = caffe::SolverParameter_SolverType_SGD;
		} else if (strcmp("Nesterov", SolverType) == 0) {
			type = caffe::SolverParameter_SolverType_NESTEROV;
		} else if (strcmp("AdaGrad", SolverType) == 0) {
			type = caffe::SolverParameter_SolverType_ADAGRAD;
		} else if (strcmp("RMSProp", SolverType) == 0) {
			type = caffe::SolverParameter_SolverType_RMSPROP;
		} else if (strcmp("AdaDelta", SolverType) == 0) {
			type = caffe::SolverParameter_SolverType_ADADELTA;
		} else if (strcmp("Adam", SolverType) == 0) {
			type = caffe::SolverParameter_SolverType_ADAM;
		}
		switch (type) {
		case caffe::SolverParameter_SolverType_SGD:
			slocal.reset(new caffe::SGDSolver<float>(param));
			break;
		case caffe::SolverParameter_SolverType_NESTEROV:
			slocal.reset(new caffe::NesterovSolver<float>(param));
			break;
		case caffe::SolverParameter_SolverType_ADAGRAD:
			slocal.reset(new caffe::AdaGradSolver<float>(param));
			break;
		case caffe::SolverParameter_SolverType_RMSPROP:
			slocal.reset(new caffe::RMSPropSolver<float>(param));
			break;
		case caffe::SolverParameter_SolverType_ADADELTA:
			slocal.reset(new caffe::AdaDeltaSolver<float>(param));
			break;
		case caffe::SolverParameter_SolverType_ADAM:
			slocal.reset(new caffe::AdamSolver<float>(param));
			break;
		default:
			LOG(FATAL) << "Unknown SolverType: " << type;
		}
		auto tlocal = slocal->net();
		printf("net: %s\n", param.net().c_str());
		// printf("release_net: %s\n",
		// param.release_net()?param.release_net()->c_str():"null");
		// printf("mutable_net: %s\n",
		// param.mutable_net()?param.mutable_net()->c_str():"null");
		std::shared_ptr<caffe::Net<float>> local(
			new caffe::Net<float>(param.net().c_str(), caffe::TEST));
		int blobs = tlocal->input_blobs().size();
		std::vector<float> state;
		if (local->input_blobs().size() > 0) {
			auto &i = local->input_blobs()[0];
			int ct = i->count();
			float *td = i->mutable_cpu_data();
			state.push_back(drand48() - 0.5);
			state.push_back(drand48() - 0.5);
			state.push_back(drand48() - 0.5);
			auto state_ = state.size();
			for (int j = 0; j < ct; ++j)
				td[j] = state[j % state_];
		}
		if (blobs > 0) {
			auto state_ = state.size();
			auto &i = tlocal->input_blobs()[0];
			int ct = i->count();
			float *td = i->mutable_cpu_data();
			for (int j = 0; j < ct; ++j)
				td[j] = state[j % state_];
		}
		if (blobs > 1) {
			auto &i = tlocal->input_blobs()[1];
			int ct = i->count();
			float *td = i->mutable_cpu_data();
			for (int j = 0; j < ct; ++j)
				td[j] = 23;
		}
		if (blobs > 2) {
			auto &i = tlocal->input_blobs()[2];
			int ct = i->count();
			float *td = i->mutable_cpu_data();
			for (int j = 0; j < ct; ++j)
				td[j] = 1.0;
		}

		SolverType = slocal->type();
		int i = tlocal->blob_by_name("input")->count();
		int o = tlocal->blob_by_name("output")->count();
		float s = 0.5;
		{
			auto ret = local->Forward(&s);
			float loss = 0;
			auto &o = ret;
			int cnt = 0;
			for (auto &i : o) {
				auto d = i->cpu_data();
				for (int j = 0; j < i->count(); ++j) {
					loss = d[j];
					printf("%d value = %f %f\n", -1, loss, s);
				}
				cnt += i->count();
			}
		}
		for (int i = 0; i < 300; ++i) {
			// printf("step: %i\n", i);
			if (blobs > 3) {
				auto &i = tlocal->input_blobs()[3];
				int ct = i->count();
				float *td = i->mutable_cpu_data();
				td[0] = 20.0;
				td[1] = 0;
				td[2] = 0.8;
				td[3] = 1.2;
			}
			slocal->Step(20);
			{
				float loss = 0;
				auto &o = tlocal->output_blobs();
				int cnt = 0;
				for (auto &i : o) {
					auto d = i->cpu_data();
					for (int j = 0; j < i->count(); ++j) {
						loss += d[j];
					}
					cnt += i->count();
				}
				loss /= cnt;
				printf("%d loss = %f\n", i, loss);
			}
			{
				Net::CopyModel(*tlocal.get(), *local.get());
				auto ret = local->Forward(&s);
				float loss = 0;
				auto &o = ret;
				int cnt = 0;
				for (auto &i : o) {
					auto d = i->cpu_data();
					for (int j = 0; j < i->count(); ++j) {
						loss = d[j];
						printf("%d value = %f %f\n", j, loss, s);
					}
					cnt += i->count();
				}
			}
		}
	}
}