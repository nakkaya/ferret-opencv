(defnative imread [n]
  (on "defined FERRET_STD_LIB"
      ("opencv2/core/core.hpp"
       "opencv2/highgui/highgui.hpp")
      "using namespace cv;
       __result = obj<value<Mat>>(imread(string::to<std::string>(n) , CV_LOAD_IMAGE_COLOR));"))

(defnative imwrite [n f]
  (on "defined FERRET_STD_LIB"
      ("opencv2/core/core.hpp"
       "opencv2/highgui/highgui.hpp")
      "using namespace cv;
       imwrite(string::to<std::string>(n) , value<Mat>::to_value(f));"))

(defnative named-window [name]
  (on "defined FERRET_STD_LIB"
      ("opencv2/core/core.hpp"
       "opencv2/highgui/highgui.hpp")
      "using namespace cv;
       namedWindow(string::to<std::string>(name), CV_WINDOW_AUTOSIZE);"))

(defnative move-window [n x y]
  (on "defined FERRET_STD_LIB"
      ("opencv2/core/core.hpp"
       "opencv2/highgui/highgui.hpp")
      "using namespace cv;
       moveWindow(string::to<std::string>(n), number::to<number_t>(x), number::to<number_t>(y));"))

(defnative imshow [n f]
  (on "defined FERRET_STD_LIB"
      ("opencv2/core/core.hpp"
       "opencv2/highgui/highgui.hpp")
      "using namespace cv;
       imshow(string::to<std::string>(n), value<Mat>::to_value(f));"))

(defnative resize [i w h]
  (on "defined FERRET_STD_LIB"
      ("opencv2/core/core.hpp"
       "opencv2/imgproc/imgproc.hpp")
      "using namespace cv;
       Size size(number::to<number_t>(w), number::to<number_t>(h));
       Mat dst;
       resize(value<Mat>::to_value(i), dst, size);
       __result = obj<value<Mat>>(dst);"))

(defn wait-key [n]
  "__result = obj<number>(cv::waitKey(number::to<number_t>(n)));")

(defnative bgr-to-hsv [i t]
  (on "defined FERRET_STD_LIB"
      ("opencv2/core/core.hpp"
       "opencv2/imgproc/imgproc.hpp")
      "using namespace cv;
       __result = obj<number>(CV_BGR2HSV);"))

(defnative cvt-color [i t]
  (on "defined FERRET_STD_LIB"
      ("opencv2/core/core.hpp"
       "opencv2/imgproc/imgproc.hpp")
      "using namespace cv;
       Mat dst;
       cvtColor(value<Mat>::to_value(i), dst, number::to<number_t>(t));
       __result = obj<value<Mat>>(dst);"))

(defnative in-range [i [min-h min-s min-v] [max-h max-s max-v]]
  (on "defined FERRET_STD_LIB"
      ("opencv2/core/core.hpp"
       "opencv2/imgproc/imgproc.hpp")
      "using namespace cv;
       Mat dst;

       inRange(value<Mat>::to_value(i),
               Scalar(number::to<number_t>(min_h), number::to<number_t>(min_s), number::to<number_t>(min_v), 0), 
               Scalar(number::to<number_t>(max_h), number::to<number_t>(max_s), number::to<number_t>(max_v), 0), 
               dst);

       __result = obj<value<Mat>>(dst);"))

(defnative video-capture [n]
  (on "defined FERRET_STD_LIB"
      ("opencv2/opencv.hpp")
      "using namespace cv;
       VideoCapture cap;

       if (n.is_type(runtime::type::number))
         cap.open(number::to<number_t>(n));
       else
         cap.open(string::to<std::string>(n));

       if (!cap.isOpened())
         return nil();

       __result = obj<value<VideoCapture>>(cap);"))

(defnative video-capture-release [c]
  (on "defined FERRET_STD_LIB"
      ("opencv2/opencv.hpp")
      "using namespace cv;
       value<VideoCapture>::to_value(c).release();"))

(defnative query-capture [c]
  (on "defined FERRET_STD_LIB"
      ("opencv2/opencv.hpp")
      "using namespace cv;
       Mat frame;
       value<VideoCapture>::to_value(c) >> frame;
       __result = obj<value<Mat>>(frame);"))
