#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    Mat im(600, 800, CV_32F);
    putText(im, "Hello, World!", Point(20,40), 0, 1, Scalar(255, 255, 255));
    if (im.empty())
    {
        cerr << "Cannot open image!" << endl;
        return -1;
    }

    imshow("image", im);
    waitKey(0);

    return 0;
}
