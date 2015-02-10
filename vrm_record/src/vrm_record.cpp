#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <memory>
#include <boost/format.hpp>

#include <VRmUsbCam.h>

#define WINDOW_NAME "Capture"
#define WHITE cv::Scalar(255, 255, 255)

using namespace cv;

int main(int argc, char** argv)
{
    std::string save_dir = argc >= 2 ? argv[1] : "../output/";

    int key = 0;
    cv::namedWindow(WINDOW_NAME);
    cv::Mat frame;

    std::shared_ptr<cv::VideoCapture> capture = std::make_shared<VRmUsbCam>();
    if (!capture->open(0))
    {
        std::cerr << "Error opening vrm camera device" << std::endl;
        std::cerr << "Try opening default camera device" << std::endl;
        capture = std::make_shared<VideoCapture>(0);
        if(!capture->isOpened())
        {
            std::cerr << "Error opening default camera device" << std::endl;
            return EXIT_FAILURE;
        }
    }

    std::cout << "VRm image and video recording" << std::endl;
    std::cout << "  Press 's' to save an image" << std::endl;
    std::cout << "  Press 'r' to start/stop video recording" << std::endl;
    std::cout << "  Press 'q' to quit" << std::endl;

    cv::VideoWriter video_out;
    int record = false;

    int cnt_frame = 0;
    int cnt_img = 0;
    int cnt_vid = 0;
    while (key != 'q')
    {
        *capture >> frame;
        cv::imshow(WINDOW_NAME, frame);
        key = cv::waitKey(1);

        if (key == 's') // save image
        {
            std::stringstream s;
            s << boost::format(save_dir + "img_%03d.png") % cnt_img;
            cv::imwrite(s.str(), frame);
            cnt_img++;

            std::cout << "Saved image: " + s.str() << std::endl;
        }

        if (key == 'r') // recording
        {
            if (!record)
            {
                double input_fps = capture->get(CV_CAP_PROP_FPS);
                if (input_fps <= 0.0)
                    input_fps = 25.0;

                std::stringstream s;
                s << boost::format(save_dir + "vid_%03d.avi") % cnt_vid;

                video_out.open(s.str(), CV_FOURCC('X', 'V', 'I', 'D'),
                               input_fps, frame.size(), false);
                if (!video_out.isOpened())
                {
                    std::cerr << "Could not write output video" << std::endl;
                    return EXIT_FAILURE;
                }

                std::cout << "Started recording video: " << s.str()
                          << std::endl;
                record = true;
            }
            else
            {
                video_out.release();
                cnt_vid++;

                std::cout << "Stopped recording video"  << std::endl;
                record = false;
            }
        }

        if (record)
        {
            video_out << frame;
        }

        cnt_frame++;
    }


    return EXIT_SUCCESS;
}
