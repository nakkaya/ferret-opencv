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
  "cv::waitKey(number::to<number_t>(n));")

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
       VideoCapture cap(number::to<number_t>(n));
       __result = obj<value<VideoCapture>>(cap);"))

(defnative query-capture [c]
  (on "defined FERRET_STD_LIB"
      ("opencv2/opencv.hpp")
      "using namespace cv;
       Mat frame;
       value<VideoCapture>::to_value(c) >> frame;
       __result = obj<value<Mat>>(frame);"))

;; (native-header "opencv/cv.h"
;;                "opencv2/core/core.hpp"
;;                "opencv2/opencv.hpp"
;;                "opencv2/highgui/highgui.hpp")

;; (defn capture-from-cam [idx]
;;   "CvCapture* capture = cvCaptureFromCAM(0);

;;    if(!capture)
;;      return nil();

;;    __result = obj<pointer>(capture);")

;; (defn query-frame [c]
;;   "CvCapture* capture = pointer::to_pointer<CvCapture>(c);
;;    IplImage* f = cvQueryFrame(capture);

;;    if (!f)
;;      return nil();
;;    __result = obj<pointer>(f);")

;; (defn capture-seq [c]
;;   (if-let [frame (query-frame c)]
;;     (cons frame (lazy-seq (capture-seq c)))))

;; (defn save-image [n f]
;;   "IplImage* frame = pointer::to_pointer<IplImage>(f);
;;    cvSaveImage(string::to<std::string>(n).c_str(), frame);")

;; (defn load-image [n]
;;   "IplImage* frame = cvLoadImage(string::to<std::string>(n).c_str());
;;    __result = obj<pointer>(frame);")

;; (defn create-file-capture [f]
;;   "CvCapture* capture = cvCreateFileCapture(string::to<std::string>(f).c_str());

;;    if(!capture)
;;      return nil();
;;    __result = obj<pointer>(capture);")

;; (defn named-window [name]
;;   "cvNamedWindow(string::to<std::string>(name).c_str(), CV_WINDOW_AUTOSIZE);")

;; (defn resize-window [n w h]
;;   "cvResizeWindow(string::to<std::string>(n).c_str(), number::to<number_t>(w), number::to<number_t>(h))")

;; (defn move-window [n x y]
;;   "cvMoveWindow(string::to<std::string>(n).c_str(), number::to<number_t>(x), number::to<number_t>(y))")

;; (defn show-image [n f]
;;   "IplImage* frame = pointer::to_pointer<IplImage>(f);
;;    cvShowImage(string::to<std::string>(n).c_str(), frame);")

;; (defn resize-image [i w h]
;;   "IplImage* source = pointer::to_pointer<IplImage>(i);
;;    IplImage* destination = cvCreateImage (cvSize(number::to<number_t>(w), number::to<number_t>(h)),
;;                                           source->depth, source->nChannels);
;;    cvResize(source, destination);
;;    __result = obj<pointer>(destination);")

;; (defn bgr-to-hsv [i]
;;   "IplImage* source = pointer::to_pointer<IplImage>(i);
;;    CvSize size = cvGetSize(source);
;;    IplImage *hsv = cvCreateImage(size, IPL_DEPTH_8U, 3);
;;    cvCvtColor(source, hsv, CV_BGR2HSV);
;;    __result = obj<pointer>(hsv);")

;; (defn in-range-s [i [min-h min-s min-v] [max-h max-s max-v]]
;;   "IplImage* source = pointer::to_pointer<IplImage>(i);
;;    CvSize size = cvGetSize(source);
;;    CvMat *mask = cvCreateMat(size.height, size.width, CV_8UC1);
;;    cvInRangeS(source, 
;;               cvScalar(number::to<number_t>(min_h), number::to<number_t>(min_s), number::to<number_t>(min_v), 0),
;;               cvScalar(number::to<number_t>(max_h), number::to<number_t>(max_s), number::to<number_t>(max_v), 0), 
;;               mask);
;;    __result = obj<pointer>(mask);")
