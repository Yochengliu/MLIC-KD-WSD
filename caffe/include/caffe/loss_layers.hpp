#ifndef CAFFE_LOSS_LAYERS_HPP_
#define CAFFE_LOSS_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class CrossEntropyLossLayer : public LossLayer<Dtype> {
 public:
  explicit CrossEntropyLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) ,
          neg_target_(new Blob<Dtype>()),
	  neg_prediction_(new Blob<Dtype>()),
	  result1_(new Blob<Dtype>()),
	  result2_(new Blob<Dtype>()),
          prediction_safe_(new Blob<Dtype>()) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CrossEntropyLoss"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype>* neg_target_;
  Blob<Dtype>* neg_prediction_;
  Blob<Dtype>* target_;
  Blob<Dtype>* prediction_;
  Blob<Dtype>* result1_;
  Blob<Dtype>* result2_; 
  Blob<Dtype>* prediction_safe_;
};

}  // namespace caffe

#endif  // CAFFE_LOSS_LAYERS_HPP_
