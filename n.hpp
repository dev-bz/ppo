#pragma once
#include <memory>
#include <vector>
namespace caffe {
template <typename Dtype> class Net;
template <typename Dtype> class Solver;
template <typename Dtype> class Blob;
} // namespace caffe
struct Net {
	std::shared_ptr<caffe::Net<float>> local;
	std::shared_ptr<caffe::Net<float>> batch;
	std::shared_ptr<caffe::Net<float>> tlocal;
	std::shared_ptr<caffe::Solver<float>> slocal;
	int GetBatchSize();
	void LoadTrainData(const std::vector<float> &X, const std::vector<float> &Y, const std::vector<float> &W);
	void SetTrainParam(const std::vector<float> &param);
        void SetTrainParam(const char *from, const char *to);
	float GetTrainResult();
	static int CopyParams(const std::vector<caffe::Blob<float> *> &src_params,
												 const std::vector<caffe::Blob<float> *> &dst_params);
	static int CopyModel(const caffe::Net<float> &src, caffe::Net<float> &dst);
	void makeNet(int i, int o, const char *file, bool solver = true);
	void syncNet(bool train = true);
        float trainNet(int iters);
	std::vector<float> &getValue(const std::vector<float> &input);
        std::vector<float> &getValue(const std::vector<float> &input,const char*second,std::vector<float> &output);
	float getValues(const std::vector<float> &input, std::vector<float> &output, int batchSize = 1024);
	void Save(const std::string &model_file);
	void Load(const std::string &model_file);
	float check(float weight_decay = 1);
	std::vector<float> local_data;
	std::vector<float> batch_data;
};
extern "C" void shutDownCaffe();
extern "C" void startupCaffe();