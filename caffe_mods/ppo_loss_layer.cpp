#include "caffe/util/math_functions.hpp"
#include "ppo_loss_layer.hpp"
#include <vector>
namespace caffe {
template <typename Dtype>
void PPOLossLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                  const vector<Blob<Dtype> *> &top) {
  CHECK_EQ(bottom[2]->shape(0), bottom[1]->shape(0))
      << "Input 2 and 1 must have the same num.";
  CHECK_EQ(bottom[2]->count(1), 1) << "Input 2 last dimension must == 1.";
  CHECK_GE(bottom[3]->count(), 4) << "Input 3 size must >=4.";
  LossLayer<Dtype>::Reshape(bottom, top);
  advantage.ReshapeLike(*bottom[0]);
  oldlogp.ReshapeLike(*bottom[0]);
  newlogp.ReshapeLike(*bottom[0]);
  prob.ReshapeLike(*bottom[0]);
  diff_.ReshapeLike(*bottom[0]);
  td.ReshapeLike(*bottom[0]);
}
/*
        bottom: "output"
        bottom: "label"
        bottom: "advantage"
        bottom: "param"
*/
/*
float NormalLogp(float scale, float x) {
        float sx = x / scale;
        return -0.5 * sx * sx - 0.5 * logf(M_PI + M_PI) - logf(scale);
}
float NormalProb(float scale, float x) {
        float sx = x / scale;
        return expf(-0.5 * sx * sx - 0.5 * logf(M_PI + M_PI) - logf(scale));
}
float NormalTD(float scale, float x) {
        return -sqrt(0.5/pi)x*e^(-0.5(x/s)^2)/(s^3)
        return -sqrt(0.5/pi) (x/s/s)*e^(-0.5 (x/s)^2)
        //return -x * expf(-0.5 * x * x - 0.5 * logf(M_PI + M_PI) -
logf(scale));
}
*/
#define DEBUG_PPO(x)
template <typename Dtype>
void PPOLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                      const vector<Blob<Dtype> *> &top) {
  int count = bottom[0]->count();
  int num = bottom[2]->count();
  DEBUG_PPO(printf("start (%f,%f,%f)\n", bottom[0]->cpu_data()[0],
                   bottom[1]->cpu_data()[0], bottom[2]->cpu_data()[0]));
  caffe_sub(count, bottom[1]->cpu_data(), bottom[0]->cpu_data(),
            this->diff_.mutable_cpu_data());

  const Dtype scale = bottom[3]->cpu_data()[0];
  Dtype &sw = bottom[3]->mutable_cpu_data()[1];
  const Dtype &low = bottom[3]->cpu_data()[2];
  const Dtype &high = bottom[3]->cpu_data()[3];
  int nan_count = 0;
#if 1
  {
    // x/s
    caffe_scal(count, Dtype(1 / scale), this->diff_.mutable_cpu_data());
    // (x/s)^2
    caffe_sqr(count, this->diff_.cpu_data(), this->newlogp.mutable_cpu_data());
    // -0.5 (x/s)^2
    caffe_scal(count, Dtype(-0.5), this->newlogp.mutable_cpu_data());
    // e ^ (-0.5 (x/s)^2)
    caffe_exp(count, this->newlogp.cpu_data(), this->td.mutable_cpu_data());
    // -0.5 (x/s)^2 - 0.5 ln(2π)
    caffe_add_scalar(count, -Dtype(0.5 * log(M_PI + M_PI) /* + log(1.0)*/),
                     this->newlogp.mutable_cpu_data());
  }
  if (sw == 0.0) {
    caffe_copy(count, this->newlogp.cpu_data(),
               this->oldlogp.mutable_cpu_data());
    // prob = e^-0.5 (x/s)^2 - 0.5 ln(2π)
    caffe_exp(count, this->oldlogp.cpu_data(), this->prob.mutable_cpu_data());
    // prob += 0.00001
    caffe_add_scalar(count, Dtype(1e-5), this->prob.mutable_cpu_data());
    // set flag for once
    sw = Dtype(1.0);
  }
  // Gradient = (-sqrt(0.5/π)x/s/s) e ^ (-0.5 (x/s)^2)
  caffe_scal(count, Dtype(-sqrt(0.5 * M_1_PI) / scale),
             this->td.mutable_cpu_data());
  caffe_mul(count, this->td.cpu_data(), this->diff_.cpu_data(),
            this->td.mutable_cpu_data());
  caffe_scal(count, Dtype(1.0 / bottom[0]->count(1)),
             this->td.mutable_cpu_data());
#endif
  caffe_div(count, this->td.cpu_data(), prob.cpu_data(),
            this->td.mutable_cpu_data());
  caffe_sub(count, this->newlogp.cpu_data(), this->oldlogp.cpu_data(),
            this->diff_.mutable_cpu_data());
  Dtype *data = this->diff_.mutable_cpu_data();
  // printf("prob[%d](%f,%f,%f) %f %f\n", 0, bottom[0]->cpu_data()[0],
  // bottom[1]->cpu_data()[0], bottom[2]->cpu_data()[0],
  // this->prob.cpu_data()[0], this->td.cpu_data()[0]);
  for (int i = 0; i < count; ++i) {
    if (5.0 < data[i])
      data[i] = 5.0;
    else if (-5.0 > data[i])
      data[i] = -5.0;
  }
  /*ratio*/ caffe_exp(count, this->diff_.cpu_data(),
                      this->diff_.mutable_cpu_data());
  const Dtype *ratio = this->diff_.cpu_data();
  /*
  caffe_exp(count, this->newlogp.cpu_data(), this->prob.mutable_cpu_data());
  caffe_div(count, this->prob.cpu_data(), ratio, this->prob.mutable_cpu_data());
  caffe_add_scalar(count, Dtype(1e-5), this->prob.mutable_cpu_data());
  caffe_div(count, this->td.cpu_data(), this->prob.cpu_data(),
  this->td.mutable_cpu_data());
  */
  const Dtype *adv = bottom[2]->cpu_data();
  int col = count / num;
  data = this->advantage.mutable_cpu_data();

  // caffe_mul(count, ratio, adv, data);
  Dtype *td = this->td.mutable_cpu_data();
  // printf("adv: %f ratio: %f\n", data[0], ratio[0]);
  Dtype overCount = 0;
  int nan_idx = 0;
  /*for (int i = 0; i < col; ++i) { printf("%.2f ", ratio[i]); }
  printf("]ratio\n");*/
  Dtype tloss = 0.0;
  for (int i = 0; i < count; ++i) {
    int ii = i / col;
    data[i] = ratio[i] * adv[ii];
    if (adv[ii] > 0 && high < ratio[i]) {
      tloss += high * adv[ii];
      td[i] = 0.0;
      ++overCount;
    } else if (adv[ii] < 0 && low > ratio[i]) {
      tloss += low * adv[ii];
      td[i] = 0.0;
      ++overCount;
    } else
      tloss += data[i];
    if (isnanf(td[i])) {
      td[i] = 0;
      nan_idx = i;
      ++nan_count;
    }

    // overCount = fmax(overCount, std::abs(td[i]));
    // tloss = std::min(tloss,this->prob.cpu_data()[i]);
  }
  if (bottom[3]->count() > 4) {
    bottom[3]->mutable_cpu_data()[4] = overCount / count;
  }
  top[0]->mutable_cpu_data()[0] = tloss / count;
  if (nan_count && sw == Dtype(1.0)) {
    sw = Dtype(2.0);
    Dtype diff =
        bottom[1]->cpu_data()[nan_idx] - bottom[0]->cpu_data()[nan_idx];
    Dtype q = sqrt(0.5 * M_1_PI);
    Dtype ds = diff / scale;
    Dtype dd = diff * diff;
    Dtype hd = -0.5 * dd;
    Dtype qd = q * diff;
    Dtype ep = exp(hd);
    Dtype ed = -qd * ep;
    Dtype ss = Dtype(1.0 / (scale * scale * scale * bottom[0]->count(1)));
    printf("\x1B[1;1f");
    printf(",diff(%f)\n,q(%f)\n,ds(%f)\n,dd(%f)\n,hd(%f)\n,qd(%f)\n,ep(%f)\n,"
           "ed(%f)\n,ss(%f)\n,prob[%f],ret(%f)",
           diff, q, ds, dd, hd, qd, ep, ed, ss, prob.cpu_data()[nan_idx],
           ed * ss / prob.cpu_data()[nan_idx]);
    printf(" nan td: %d/%d as (%f, %d) %f\n", nan_count, count, scale,
           bottom[0]->count(1), ss);
  }
  DEBUG_PPO(printf("  step loss %f, %f\n\n", tloss / count, td[0]));
}
template <typename Dtype>
void PPOLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                       const vector<bool> &propagate_down,
                                       const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[0])
    caffe_mul(bottom[0]->count(), this->td.cpu_data(),
              this->advantage.cpu_data(), bottom[0]->mutable_cpu_diff());
}
#ifdef CPU_ONLY
STUB_GPU(PPOLossLayer);
#endif
INSTANTIATE_CLASS(PPOLossLayer);
REGISTER_LAYER_CLASS(PPOLoss);
} // namespace caffe