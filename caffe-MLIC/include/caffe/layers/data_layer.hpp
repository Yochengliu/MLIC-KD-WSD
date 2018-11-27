#ifndef CAFFE_DATA_LAYER_HPP_
#define CAFFE_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template <typename Dtype>
class DataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit DataLayer(const LayerParameter& param);
  virtual ~DataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "Data"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);

  DataReader reader_;
};

template <typename Dtype>
class HumanAttDataLayer : public BasePrefetchingDataLayer_v2<Dtype> {
 public:
  explicit HumanAttDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer_v2<Dtype>(param) {}
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

}  // namespace caffe

#endif  // CAFFE_DATA_LAYER_HPP_
