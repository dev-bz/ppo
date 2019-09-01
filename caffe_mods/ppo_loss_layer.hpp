#ifndef CAFFE_PPO_LOSS_LAYER_HPP_
#define CAFFE_PPO_LOSS_LAYER_HPP_

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class PPOLossLayer : public LossLayer<Dtype> {
 public:
  explicit PPOLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), oldlogp(), newlogp(), prob(), td(), diff_(), advantage() {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 4; }
  virtual inline const char* type() const { return "PPOLoss"; }
 

 protected:
  /// @copydoc EuclideanLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	Blob<Dtype> oldlogp,newlogp,prob,td;
	Blob<Dtype> advantage;
	Blob<Dtype> diff_;
};

}  // namespace caffe

#endif  // CAFFE_WEIGHTED_EUCLIDEAN_LOSS_LAYER_HPP_