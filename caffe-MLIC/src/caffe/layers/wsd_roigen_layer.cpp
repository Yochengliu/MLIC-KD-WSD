#include <vector>

#include "caffe/common_layers.hpp"

namespace caffe {

template <typename Dtype>
void WsdRoigenLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //roi_num = this->layer_param_.wsd_roigen_param().roi_num();
  //LOG(INFO) << "WsdRoigenLayer Setup Done!";
}

template <typename Dtype>
void WsdRoigenLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
//  LOG(INFO) << "WsdRoigenlayer Reshape start";
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
//  LOG(INFO) << "WsdRoigenlayer Reshape Done!";
}

template <typename Dtype>
void WsdRoigenLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
//  LOG(INFO) << "WsdRoigen Layer Forward Start";
  //LOG(INFO) << "top " << top[0]->num() << top[0]->channels() << top[0]->height() << top[0]->width();
  float tmp_roi_num = 0.0;
  int roi_num = 0;
  tmp_roi_num = bottom[0]->channels()/5.0;
  CHECK_GT(tmp_roi_num, 0.0);
  if ((tmp_roi_num - int(tmp_roi_num)) < 1e-10) {
	roi_num = int(tmp_roi_num);
  }
  else {
    LOG(ERROR) << "The RoI cordinates number is not compatible with 4: " << bottom[0]->channels();
  }
  
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  // LOG(ERROR) << "bottomsize " << bottom[0]->num();
  //CHECK_EQ(bottom[0]->channels(), roi_num*4);

  for (int i = 0; i < bottom[0]->num(); ++i) {
    for (int j = 0; j < roi_num; ++j)
    {
      top_data[6*(i*roi_num+j) + 0] =  i;  // batchindex
      top_data[6*(i*roi_num+j) + 1] = bottom_data[bottom[0]->channels() * i+ 5 *j + 0];  // xmin
      top_data[6*(i*roi_num+j) + 2] = bottom_data[bottom[0]->channels() * i+ 5 *j + 1];  // ymin
      top_data[6*(i*roi_num+j) + 3] = bottom_data[bottom[0]->channels() * i+ 5 *j + 2];  // xmax
      top_data[6*(i*roi_num+j) + 4] = bottom_data[bottom[0]->channels() * i+ 5 *j + 3];  // ymax
      top_data[6*(i*roi_num+j) + 5] = bottom_data[bottom[0]->channels() * i+ 5 *j + 4];  // score
    }
  }
//  LOG(INFO) << "WsdRoigen Layer Forward Done!";
}

INSTANTIATE_CLASS(WsdRoigenLayer);
REGISTER_LAYER_CLASS(WsdRoigen);

}  // namespace caffe
