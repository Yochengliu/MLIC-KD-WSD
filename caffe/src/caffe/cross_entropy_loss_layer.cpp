#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "CROSS_ENTROPY_LOSS layer inputs must have the same count.";
 
  neg_target_->ReshapeLike(*bottom[0]);
  neg_prediction_->ReshapeLike(*bottom[0]); 
  result1_->ReshapeLike(*bottom[0]);
  result2_->ReshapeLike(*bottom[0]);
  prediction_safe_->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss = 0;

  for (int i = 0; i < count; ++i) {
    if (target[i]==1) {
      loss -= log(std::max(input_data[i],Dtype(0.0001)));
    }
    else {
      loss -= log(std::max(1-input_data[i],Dtype(0.0001)));
    }
  }

  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* target = bottom[1]->cpu_data();
    const Dtype* prediction = bottom[0]->cpu_data();
    Dtype* neg_target = neg_target_->mutable_cpu_data();
    Dtype* neg_prediction = neg_prediction_->mutable_cpu_data();
    Dtype* result1 = result1_->mutable_cpu_data();
    Dtype* result2 = result2_->mutable_cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    Dtype* prediction_safe = prediction_safe_->mutable_cpu_data();
    caffe_copy(count, target, neg_target); 
    caffe_copy(count, prediction, neg_prediction);
    caffe_scal(count, Dtype(-1), neg_target);
    caffe_add_scalar(count, Dtype(1), neg_target);
    caffe_add_scalar(count, Dtype(-1), neg_prediction);
    for (int tmp_cnt = 0; tmp_cnt < count; ++tmp_cnt) {
      prediction_safe[tmp_cnt] = std::max(prediction[tmp_cnt],Dtype(0.0001));
    }
     for (int tmp_cnt = 0; tmp_cnt < count; ++tmp_cnt) {
      neg_prediction[tmp_cnt] = std::min(neg_prediction[tmp_cnt],Dtype(-0.0001));
    }
    caffe_div(count, target, prediction_safe, result1);
    caffe_div(count, neg_target, neg_prediction, result2);
    caffe_add(count, result1, result2, bottom_diff);
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, -loss_weight / num, bottom_diff);
  }
}

INSTANTIATE_CLASS(CrossEntropyLossLayer);
REGISTER_LAYER_CLASS(CrossEntropyLoss);

}  // namespace caffe
