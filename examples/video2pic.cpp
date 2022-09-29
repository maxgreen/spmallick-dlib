//
// Created by ganho on 2022/9/29.
/**
 * 视频抽帧demo
 */

#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
using namespace std;


int main()
{
    try
    {
        cv::VideoCapture cap("ghj_2022-09-29.mp4");
        bool suc = cap.isOpened();
        int fps = int(cap.get(cv::CAP_PROP_FPS));  // 获取帧率
        int width = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));  // 获取宽度
        int height = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT)); // 获取高度
        int frame_count =0;
        while(suc){

            cv::Mat frame_key;
            suc = cap.read(frame_key);
            if(!suc){
                break;
            }
            else{
                std::string str1 = std::string("pic/").append(std::to_string(frame_count).append(".jpg"));
                cv::imwrite(str1, frame_key);
                frame_count += 1;
            }
            cv::waitKey(1);
        }
        cap.release();
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
}

