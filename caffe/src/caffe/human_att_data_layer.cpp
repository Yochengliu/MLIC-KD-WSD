#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/random.hpp>

#include <fstream>  
#include <iostream> 
#include <string>
#include <utility>
#include <vector>
#include <cmath>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/mpi_function.hpp"

namespace caffe {

template <typename Dtype>
HumanAttDataLayer<Dtype>::~HumanAttDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
int HumanAttDataLayer<Dtype>::Rand(int n) {
  CHECK(scale_rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(scale_rng_->generator());
  return ((*rng)() % n);
}

template <typename Dtype>
void HumanAttDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.human_att_data_param().new_height();
  const int new_width  = this->layer_param_.human_att_data_param().new_width();
  const bool is_color  = this->layer_param_.human_att_data_param().is_color();
  string root_folder = this->layer_param_.human_att_data_param().root_folder();
  const int att_num = this->layer_param_.human_att_data_param().att_num();
  const int bbox_num = this->layer_param_.human_att_data_param().bbox_num();
  
  const int thread_id = Caffe::getThreadId();
  const int thread_num = Caffe::getThreadNum();
  const int batch_size = this->layer_param_.human_att_data_param().batch_size() / thread_num;
  CHECK(batch_size != 0) << "Batch size configured in model prototxt is less than the number of GPUs";
  
  int label_dim = att_num + bbox_num * 5;

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  const string& source = this->layer_param_.human_att_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());

  string line = "";
  if(infile.is_open()) {
    while(!infile.eof()) {
      getline(infile, line);
	  if(line != "") {
      vector<string> line_strlist = StrSplit(line, " ");
      CHECK(line_strlist.size() == (label_dim + 1)) << "Format error '" << line << "', or wrong lable_dim was set.";
      string img_path = line_strlist.at(0);

      vector<float> labels;
      for (int i = 0; i < label_dim; i++)
      {
        labels.push_back(std::atof(line_strlist.at(1 + i).c_str()));
      }

      lines_.push_back(std::make_pair(img_path, labels));
	  }
    }
  }
  
  if (this->layer_param_.human_att_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";
  
  lines_id_ = lines_.size() / thread_num * thread_id;
  
  LOG(INFO) << "Layer set up lines_id_: " << lines_id_;
  
  if (this->layer_param_.human_att_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.human_att_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size() / thread_num, skip) << "Not enough points to skip";
    lines_id_ += skip;
  }
  
  //Check if we want to do random scaling
  if (this->layer_param_.human_att_data_param().scale_factors_size() > 0) {
    for (int i = 0; i < this->layer_param_.human_att_data_param().scale_factors_size(); ++i) {
        scale_factors_.push_back(this->layer_param_.human_att_data_param().scale_factors(i));
    }
    const unsigned int scale_rng_seed = caffe_rng_rand();
    caffe_mpi_bcast<int>((void*)&scale_rng_seed, 1, MPI_INT);
    scale_rng_.reset(new Caffe::RNG(scale_rng_seed));
  }
  
  for(int i = 0; i < batch_size; ++i) {
    shared_ptr<DataTransformer<Dtype> > transformer(new DataTransformer<Dtype>(
                                                  this->transform_param_, this->phase_));
    transformer->InitRand();
    this->data_transformer_vec_.push_back(transformer);
  }
  
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                new_height, new_width, is_color);
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  for(int i = 0; i < batch_size; ++i) {
    this->transformed_data_vec_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(top_shape)));
  }
  
  // Reshape prefetch_data and top[0] according to the batch_size.
  top_shape[0] = batch_size;
  this->prefetch_data_.Reshape(top_shape);
  top[0]->ReshapeLike(this->prefetch_data_);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  //label
  vector<int> label_shape(2, batch_size);
  label_shape[1] = label_dim;
  top[1]->Reshape(label_shape);
  this->prefetch_label_.Reshape(label_shape);
}

template <typename Dtype>
void HumanAttDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void HumanAttDataLayer<Dtype>::InternalThreadEntry() {
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());
  HumanAttDataParameter human_att_data_param = this->layer_param_.human_att_data_param();
  const int thread_num = Caffe::getThreadNum();
  const int batch_size = human_att_data_param.batch_size() / thread_num;
  const int new_height = human_att_data_param.new_height();
  const int new_width = human_att_data_param.new_width();
  const bool is_color = human_att_data_param.is_color();
  string root_folder = human_att_data_param.root_folder();
  const int att_num = human_att_data_param.att_num();
  const int bbox_num = human_att_data_param.bbox_num();
  
  int label_dim = att_num + bbox_num * 5;

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
      new_height, new_width, is_color);
  // choose a scale to set for the image and label in one miniBatch
  Dtype scale = 0.0;
  cv::Mat cv_resized_img;
  if (scale_factors_.size() > 0) {
    int scale_ind = Rand(scale_factors_.size());
    scale = scale_factors_[scale_ind];
    if (scale != Dtype(1)) {
        int img_height = cv_img.rows;
        int img_width = cv_img.cols;
        img_height = int(img_height * scale);
        img_width = int(img_width * scale);
        cv::resize(cv_img, cv_resized_img, cv::Size(img_width, img_height), 0, 0, cv::INTER_LINEAR);
    } else {
        cv_resized_img = cv_img;
    }
  } else {
    cv_resized_img = cv_img;
  }    
  
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_resized_img);
  this->transformed_data_.Reshape(top_shape);
  for(int i = 0; i < this->transformed_data_vec_.size(); i++) {
    this->transformed_data_vec_[i]->Reshape(top_shape);
  }
  // Reshape prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  this->prefetch_data_.Reshape(top_shape);

  Dtype* prefetch_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* prefetch_label = this->prefetch_label_.mutable_cpu_data();
  
  const int lines_size = lines_.size();
  
  if ((lines_id_ + batch_size - 1 >= lines_size) || (batch_size <= 2)) {
        for (int item_id = 0; item_id < batch_size; ++item_id) {
            CHECK_GT(lines_size, lines_id_);
            for(int i=0; i<label_dim; ++i) {
                prefetch_label[label_dim * item_id + i] = lines_[lines_id_].second.at(i);
            }
            
            cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                new_height, new_width, is_color);
            CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
            
            cv::Mat cv_resized_img;
              if (scale_factors_.size() > 0) {
                if (scale != Dtype(1)) {
                    int img_height = cv_img.rows;
                    int img_width = cv_img.cols;
                    img_height = int(img_height * scale);
                    img_width = int(img_width * scale);
                    cv::resize(cv_img, cv_resized_img, cv::Size(img_width, img_height), 0, 0, cv::INTER_LINEAR);
                    // label transformation
                    for (int i = 0; i < bbox_num; ++i) {
                        int xmin = prefetch_label[label_dim * item_id + att_num + i*5 + 0];
                        int ymin = prefetch_label[label_dim * item_id + att_num + i*5 + 1];
                        int xmax = prefetch_label[label_dim * item_id + att_num + i*5 + 2];
                        int ymax = prefetch_label[label_dim * item_id + att_num + i*5 + 3];
                        xmin = int(xmin * scale);
                        ymin = int(ymin * scale);
                        xmax = int(xmax * scale);
                        ymax = int(ymax * scale);
                        prefetch_label[label_dim * item_id + att_num + i*5 + 0] = xmin;
                        prefetch_label[label_dim * item_id + att_num + i*5 + 1] = ymin;
                        prefetch_label[label_dim * item_id + att_num + i*5 + 2] = xmax;
                        prefetch_label[label_dim * item_id + att_num + i*5 + 3] = ymax;
                    }
                }
                else {
                    cv_resized_img = cv_img;
                }
              } else {
                cv_resized_img = cv_img;
              }    
            
            int offset = this->prefetch_data_.offset(item_id);
            this->transformed_data_.set_cpu_data(prefetch_data + offset);
            this->data_transformer_->TransformDataLabel(cv_resized_img, &(this->transformed_data_), item_id, att_num, bbox_num, prefetch_label);
            // go to the next iter
            lines_id_++;
            if (lines_id_ >= lines_size) {
              // We have reached the end. Restart from the first.
              LOG(INFO) << "Restarting data prefetching from start.";
              lines_id_ = 0;
              if (this->layer_param_.human_att_data_param().shuffle()) {
                ShuffleImages();
              }
            }
        }
  } else {
    // normal batch use omp to prefetch
    #pragma omp parallel for
    for (int item_id = 0; item_id < batch_size; ++item_id) {
        int img_id = lines_id_ + item_id;
        
        for(int i=0; i<label_dim; ++i) {
            prefetch_label[label_dim * item_id + i] = lines_[img_id].second.at(i);
        }
        
        cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[img_id].first,
            new_height, new_width, is_color);
        CHECK(cv_img.data) << "Could not load " << lines_[img_id].first;
        
        cv::Mat cv_resized_img;
          if (scale_factors_.size() > 0) {
            if (scale != Dtype(1)) {
                int img_height = cv_img.rows;
                int img_width = cv_img.cols;
                img_height = int(img_height * scale);
                img_width = int(img_width * scale);
                cv::resize(cv_img, cv_resized_img, cv::Size(img_width, img_height), 0, 0, cv::INTER_LINEAR);
                // label transformation
                for (int i = 0; i < bbox_num; ++i) {
                    int xmin = prefetch_label[label_dim * item_id + att_num + i*5 + 0];
                    int ymin = prefetch_label[label_dim * item_id + att_num + i*5 + 1];
                    int xmax = prefetch_label[label_dim * item_id + att_num + i*5 + 2];
                    int ymax = prefetch_label[label_dim * item_id + att_num + i*5 + 3];
                    xmin = int(xmin * scale);
                    ymin = int(ymin * scale);
                    xmax = int(xmax * scale);
                    ymax = int(ymax * scale);
                    prefetch_label[label_dim * item_id + att_num + i*5 + 0] = xmin;
                    prefetch_label[label_dim * item_id + att_num + i*5 + 1] = ymin;
                    prefetch_label[label_dim * item_id + att_num + i*5 + 2] = xmax;
                    prefetch_label[label_dim * item_id + att_num + i*5 + 3] = ymax;
                }
            }
            else {
                cv_resized_img = cv_img;
            }
          } else {
            cv_resized_img = cv_img;
          }
          
        // Apply transformations (mirror, crop...) to the image
        int offset = this->prefetch_data_.offset(item_id);
        this->transformed_data_vec_[item_id]->set_cpu_data(prefetch_data + offset);
        this->data_transformer_vec_[item_id]->TransformDataLabel(cv_resized_img, this->transformed_data_vec_[item_id].get(), item_id, att_num, bbox_num, prefetch_label); 
    }
    
    lines_id_ += batch_size;
  }
  
  // only the thread which whole small batch exceed the dataset will shuffle here
  if (lines_id_ >= lines_size) {
    // We have reached the end. Restart from the first.
    LOG(INFO) << "Restarting data prefetching from start.";
    lines_id_ = lines_id_ % lines_size;
    if (this->layer_param_.human_att_data_param().shuffle()) {
      ShuffleImages();
    }
  }
  
}

template <typename Dtype>
std::vector<std::string> HumanAttDataLayer<Dtype>::StrSplit(std::string &str, std::string sep = ",")
{
  vector<std::string> ret_;
  if (str.empty())
  {
    return ret_;
  }

  string tmp;
  string::size_type pos_begin = str.find_first_not_of(sep);
  string::size_type comma_pos = 0;
  while (pos_begin != string::npos)
  {
    comma_pos = str.find(sep, pos_begin);
    if (comma_pos != string::npos)
    {
      tmp = str.substr(pos_begin, comma_pos - pos_begin);
      pos_begin = comma_pos + sep.length();
    }
    else
    {
      tmp = str.substr(pos_begin);
      pos_begin = comma_pos;
    }
    if (!tmp.empty())
    {
      ret_.push_back(tmp);
      tmp.clear();
    }
  }
  return ret_;
}

INSTANTIATE_CLASS(HumanAttDataLayer);
REGISTER_LAYER_CLASS(HumanAttData);

}  // namespace caffe
