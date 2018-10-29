#ifndef CAFFE_DATA_TRANSFORMER_HPP
#define CAFFE_DATA_TRANSFORMER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class DataTransformer {
 public:
  explicit DataTransformer(const TransformationParameter& param, Phase phase);
  virtual ~DataTransformer() {}
  
  void InitRand();
  
  void TransformDataLabel(const cv::Mat& cv_img,
                                       Blob<Dtype>* transformed_blob,
                                       int item_id,
                                       int att_num,
                                       int bbox_num,
                                       Dtype* prefetch_label);

 protected:
 
  virtual int Rand(int n);
 
  // Tranformation parameters
  TransformationParameter param_;

  shared_ptr<Caffe::RNG> rng_;
  Phase phase_;
  Blob<Dtype> data_mean_;
  vector<Dtype> mean_values_;
  vector<Dtype> std_values_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_TRANSFORMER_HPP_

