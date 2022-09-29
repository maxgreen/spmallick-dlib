//
// Created by ganho on 2022/9/29.
//
/**
 * 判断图片中头部的姿态位置demo
 */

#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include "render_face.hpp"

using namespace dlib;
using namespace std;

#define FACE_DOWNSAMPLE_RATIO 4
#define SKIP_FRAMES 2
#define OPENCV_FACE_RENDER


std::vector<cv::Point3d> get_3d_model_points() {
    std::vector<cv::Point3d> modelPoints;

    modelPoints.push_back(cv::Point3d(0.0f, 0.0f, 0.0f)); //The first must be (0,0,0) while using POSIT
    modelPoints.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));
    modelPoints.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));
    modelPoints.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));
    modelPoints.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f));
    modelPoints.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));

    return modelPoints;

}

std::vector<cv::Point2d> get_2d_image_points(full_object_detection &d) {
    std::vector<cv::Point2d> image_points;
    image_points.push_back(cv::Point2d(d.part(30).x(), d.part(30).y()));    // Nose tip
    image_points.push_back(cv::Point2d(d.part(8).x(), d.part(8).y()));      // Chin
    image_points.push_back(cv::Point2d(d.part(36).x(), d.part(36).y()));    // Left eye left corner
    image_points.push_back(cv::Point2d(d.part(45).x(), d.part(45).y()));    // Right eye right corner
    image_points.push_back(cv::Point2d(d.part(48).x(), d.part(48).y()));    // Left Mouth corner
    image_points.push_back(cv::Point2d(d.part(54).x(), d.part(54).y()));    // Right mouth corner
    return image_points;

}

cv::Mat get_camera_matrix(float focal_length, cv::Point2d center) {
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1);
    return camera_matrix;
}


int main(int argc, char **argv) {
    try {
        // Read input image
        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
        for (int ii = 0; ii < 314; ii++) {
            std::string fileName = std::string("pic/").append(std::to_string(ii)).append(".jpg");
            cv::Mat im = cv::imread(fileName);

            cv::Mat im_small, im_display;
            cv::resize(im, im_small, cv::Size(), 1.0 / FACE_DOWNSAMPLE_RATIO, 1.0 / FACE_DOWNSAMPLE_RATIO);
            cv::resize(im, im_display, cv::Size(), 0.5, 0.5);
            cv::Size size = im.size();
            // Change to dlib's image format. No memory is copied.
            cv_image<bgr_pixel> cimg_small(im_small);
            cv_image<bgr_pixel> cimg(im);

            // Detect faces
            std::vector<rectangle> faces = detector(cimg);
            // Pose estimation
            std::vector<cv::Point3d> model_points = get_3d_model_points();
            // Find the pose of each face.
            std::vector<full_object_detection> shapes;
            for (unsigned long i = 0; i < faces.size(); ++i) {
                rectangle r(
                        (long) (faces[i].left()),
                        (long) (faces[i].top()),
                        (long) (faces[i].right()),
                        (long) (faces[i].bottom())
                );
                full_object_detection shape = pose_model(cimg, r);
                shapes.push_back(shape);
                render_face(im, shape);
                std::vector<cv::Point2d> image_points = get_2d_image_points(shape);
                double focal_length = im.cols;
                cv::Mat camera_matrix = get_camera_matrix(focal_length, cv::Point2d(im.cols / 2, im.rows / 2));
                cv::Mat rotation_vector;
                cv::Mat rotation_matrix;
                cv::Mat translation_vector;


                cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type);

                cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector,
                             translation_vector);

                //cv::Rodrigues(rotation_vector, rotation_matrix);

                cv::Mat rotM = cv::Mat::eye(3,3,CV_64F);
                cv::Mat rotT = cv::Mat::eye(3,3,CV_64F);
                Rodrigues(rotation_vector, rotM);  //将旋转向量变换成旋转矩阵
                Rodrigues(translation_vector, rotT);

                //计算相机旋转角
                double theta_x, theta_y,theta_z;
                double PI = 3.14;
                theta_x = atan2(rotM.at<double>(2, 1), rotM.at<double>(2, 2));
                theta_y = atan2(-rotM.at<double>(2, 0),
                                sqrt(rotM.at<double>(2, 1)*rotM.at<double>(2, 1) + rotM.at<double>(2, 2)*rotM.at<double>(2, 2)));
                theta_z = atan2(rotM.at<double>(1, 0), rotM.at<double>(0, 0));
                theta_x = theta_x * (180 / PI);
                theta_y = theta_y * (180 / PI);
                theta_z = theta_z * (180 / PI);

                //计算深度
                cv::Mat P;
                P = (rotM.t()) * translation_vector;

                //输出
                cout<<ii<<endl;
                cout<<"角度"<<endl;
                cout<<theta_x<<endl;
                cout<<theta_y<<endl;
                cout<<theta_z<<endl;
                cout<<P<<endl;



                std::vector<cv::Point3d> nose_end_point3D;
                std::vector<cv::Point3d> nose_end_point3D2;
                std::vector<cv::Point2d> nose_end_point2D;
                std::vector<cv::Point2d> nose_end_point2D2;
                nose_end_point3D.push_back(cv::Point3d(0, 0, 1000.0));
                nose_end_point3D2.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));
                // objectPoints：模型坐标系下的点坐标数据
                //     可以是矩阵，单通道时，尺寸为3✖️N或者N✖️3，3通道时尺寸为1✖️N或者N✖️1
                //     可以是vector<Point3f>
                // rvec：旋转向量，Rodrigues形式，即向量的方向表示旋转轴，向量的模表示旋转弧度
                // tvec：平移向量
                // cameraMatrix：相机内参矩阵，尺寸为3✖️3
                // distCoeffs：扭曲系数，元素数为4、5或者8的向量，或者传入cv::noArray()
                // imagePoints：图像坐标系中的对应点坐标
                //     数据格式和objectPoints数据格式有对应关系，单位为像素
                //     可以是矩阵，单通道时，尺寸为2✖️N或者N✖️2，双通道时尺寸为1✖️N或者N✖️1
                //     可以是vector<Point2f>
                // jacobian：偏导数矩阵，下文详细介绍
                //     可以传cv::noArray()，或者尺寸为2N✖️(10+nDistCoeff)的矩阵
                // aspectRatio：缩放系数，即fx/fy的比值
                cv::projectPoints(nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs,
                                  nose_end_point2D);
                cv::projectPoints(nose_end_point3D2, rotation_vector, translation_vector, camera_matrix, dist_coeffs,
                                  nose_end_point2D2);
//                cv::Point2d projected_point = find_projected_point(rotation_matrix, translation_vector, camera_matrix, cv::Point3d(0,0,1000.0));
                cv::line(im, image_points[0], nose_end_point2D[0], cv::Scalar(255, 0, 0), 2);
                cv::line(im, image_points[1], nose_end_point2D2[0], cv::Scalar(255, 0, 0), 2);
//                cv::line(im,image_points[0], projected_point, cv::Scalar(0,0,255), 2);




                // Resize image for display
                im_display = im;
                cv::resize(im, im_display, cv::Size(), 0.5, 0.5);
                cv::imshow("Fast Facial Landmark Detector", im);
                std::string fileName2 = std::string("pic/").append(std::to_string(ii)).append("_1.jpg");
                cv::imwrite(fileName2, im);
            }
        }
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    catch (serialization_error &e) {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch (exception &e) {
        cout << e.what() << endl;
    }
}