#ifndef CAFFE_DATA_LAYERS_HPP_
#define CAFFE_DATA_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "boost/scoped_ptr.hpp"
#include "hdf5.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/filler.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

template <typename Dtype>
class HumanAttDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit HumanAttDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~HumanAttDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "HumanAttData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  shared_ptr<Caffe::RNG> scale_rng_;
  virtual void ShuffleImages();
  virtual int Rand(int n);
  virtual void InternalThreadEntry();

  vector<std::pair<std::string, vector<float> > > lines_;
  int lines_id_;
  std::vector<std::string> StrSplit(std::string &str, std::string sep);
  vector<Dtype> scale_factors_;
  vector<shared_ptr<DataTransformer<Dtype> > > data_transformer_vec_;
  vector<shared_ptr<Blob<Dtype> > > transformed_data_vec_;
};

}
#endif
