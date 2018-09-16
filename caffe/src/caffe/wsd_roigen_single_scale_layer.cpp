#include <vector>

#include "caffe/common_layers.hpp"

using std::floor;

namespace caffe {

template <typename Dtype>
void WsdRoigenSingleScaleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //roi_num = this->layer_param_.wsd_roigen_param().roi_num();
  //LOG(INFO) << "WsdRoigenSingleScaleLayer Setup Done!";
  single_scale_ = this->layer_param_.roi_pooling_param().single_scale();
  origin_scale_ = this->layer_param_.roi_pooling_param().origin_scale();
}

template <typename Dtype>
void WsdRoigenSingleScaleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
//  LOG(INFO) << "WsdRoigenSingleScalelayer Reshape start";
  float tmp_roi_num = 0.0;
  int roi_num = 0;
  vector<int> top_shape = bottom[0]->shape();
  tmp_roi_num = bottom[0]->channels()/5.0;
  CHECK_GT(tmp_roi_num, 0.0);
  if ((tmp_roi_num - int(tmp_roi_num)) < 1e-10) {
	roi_num = int(tmp_roi_num);
  }
  else {
    LOG(ERROR) << "The RoI cordinates number is not compatible with 4: " << bottom[0]->channels();
  }
  
  top_shape[0] = top_shape[0]*roi_num;
  top_shape[1] = 6; // roi idx: batchindex xmin ymin xmax ymax score   (zero-based) 
//  LOG(INFO) << "Top shape: " << top_shape[0] << ' ' << top_shape[1] << ' ' << top_shape[2] << ' ' << top_shape[3];
  top[0]->Reshape(top_shape);
//  LOG(INFO) << "WsdRoigenSingleScalelayer Reshape Done!";
}

template <typename Dtype>
void WsdRoigenSingleScaleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
//  LOG(INFO) << "WsdRoigenSingleScale Layer Forward Start";
  //LOG(INFO) << "top " << top[0]->num() << top[0]->channels() << top[0]->height() << top[0]->width();
  float tmp_roi_num = 0.0;
  int roi_num = 0;
  tmp_roi_num = bottom[0]->channels()/5.0;
  CHECK_GT(tmp_roi_num, 0.0);
  if ((tmp_roi_num - int(tmp_roi_num)) < 1e-10) {
	roi_num = int(tmp_roi_num);
  }
  else {
    LOG(ERROR) << "The RoI cordinates number is not compatible with 5: " << bottom[0]->channels();
  }
  
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  // LOG(ERROR) << "bottomsize " << bottom[0]->num();
  //CHECK_EQ(bottom[0]->channels(), roi_num*4);
  
  Dtype x1 = 0;
  Dtype y1 = 0;
  Dtype x2 = 0;
  Dtype y2 = 0;
  for (int i = 0; i < bottom[0]->num(); ++i) {
    for (int j = 0; j < roi_num; ++j)
    {
      x1 = bottom_data[bottom[0]->channels() * i+ 5 *j + 0];  // xmin
      y1 = bottom_data[bottom[0]->channels() * i+ 5 *j + 1];  // ymin
      x2 = bottom_data[bottom[0]->channels() * i+ 5 *j + 2];  // xmax
      y2 = bottom_data[bottom[0]->channels() * i+ 5 *j + 3];  // ymax
      top_data[6*(i*roi_num+j) + 0] =  i;  // batchindex
      top_data[6*(i*roi_num+j) + 1] = floor(Dtype(single_scale_) * x1 / Dtype(origin_scale_));
      top_data[6*(i*roi_num+j) + 2] = floor(Dtype(single_scale_) * y1 / Dtype(origin_scale_));
      top_data[6*(i*roi_num+j) + 3] = floor(Dtype(single_scale_) * x2 / Dtype(origin_scale_));
      top_data[6*(i*roi_num+j) + 4] = floor(Dtype(single_scale_) * y2 / Dtype(origin_scale_));
      top_data[6*(i*roi_num+j) + 5] = bottom_data[bottom[0]->channels() * i+ 5 *j + 4];  // score
    }
  }
//  LOG(INFO) << "WsdRoigenSingleScale Layer Forward Done!";
}

INSTANTIATE_CLASS(WsdRoigenSingleScaleLayer);
REGISTER_LAYER_CLASS(WsdRoigenSingleScale);

}  // namespace caffe
