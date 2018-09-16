#include <opencv2/core/core.hpp>
#include <boost/random.hpp>

#include <string>
#include <vector>
#include <cmath>

#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe {

template<typename Dtype>
void DataTransformer<Dtype>::TransformDataLabel(const cv::Mat& cv_img,
                                       Blob<Dtype>* transformed_blob,
                                       int item_id,
                                       int att_num,
                                       int bbox_num,
                                       Dtype* prefetch_label) {
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;
  const bool is_random_sized_crop = param_.random_sized_crop();
  int label_dim = att_num + bbox_num * 5;

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, img_channels);
  if (!is_random_sized_crop) {
    CHECK_LE(height, img_height);
    CHECK_LE(width, img_width);
  }
  CHECK_GE(num, 1);

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);
  if (!is_random_sized_crop) {
    CHECK_GE(img_height, crop_size);
    CHECK_GE(img_width, crop_size);
  }

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
     "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    if (phase_ == TRAIN) {
        h_off = Rand(img_height - crop_size + 1);
        w_off = Rand(img_width - crop_size + 1);
        cv::Rect roi(w_off, h_off, crop_size, crop_size);
        cv_cropped_img = cv_img(roi);
    } else {
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
      cv::Rect roi(w_off, h_off, crop_size, crop_size);
      cv_cropped_img = cv_img(roi);
    }
  } else {
    CHECK_EQ(img_height, height);
    CHECK_EQ(img_width, width);
  }

  CHECK(cv_cropped_img.data);

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_data[top_index] =
            (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }
    }
  }
  
  // label transformation
  if (do_mirror) {
      for (int i = 0; i < bbox_num; ++i) {
        int xmin = prefetch_label[label_dim * item_id + att_num + i*5 + 0];
        int ymin = prefetch_label[label_dim * item_id + att_num + i*5 + 1];
        int xmax = prefetch_label[label_dim * item_id + att_num + i*5 + 2];
        int ymax = prefetch_label[label_dim * item_id + att_num + i*5 + 3];
        int txmin = img_width - xmax - 1;
        int tymin = ymin;
        int txmax = img_width - xmin - 1;
        int tymax = ymax;
        prefetch_label[label_dim * item_id + att_num + i*5 + 0] = txmin;
        prefetch_label[label_dim * item_id + att_num + i*5 + 1] = tymin;
        prefetch_label[label_dim * item_id + att_num + i*5 + 2] = txmax;
        prefetch_label[label_dim * item_id + att_num + i*5 + 3] = tymax;
      }
  }
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
