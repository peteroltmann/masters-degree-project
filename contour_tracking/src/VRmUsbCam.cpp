#include "VRmUsbCam.h"

#include <iostream>
#include <signal.h>

#include <opencv2/imgproc/imgproc.hpp>

VRmUsbCam::VRmUsbCam() :
    device(0),
    port(0),
    opened(false)
{}

VRmUsbCam::~VRmUsbCam()
{
    if (opened)
        release();
}

// TODO: param ignored for now, just for opencv - take first suitable
bool VRmUsbCam::open(int deviceNo)
{
    initialize();

    if (opened)
    {
        std::cerr << "Camera device already opened" << std::endl;
        return false;
    }

    device = 0;
    port = 0;

//    // read libversion (for informational purposes only)
//    VRmDWORD libversion;
//    if (!VRmUsbCamGetVersion(&libversion))
//        return logFailure();

//    std::cout << "VRm Libversion: " << libversion << std::endl;

    // check for connected devices
    VRmDWORD size = 0;
    if (!VRmUsbCamGetDeviceKeyListSize(&size))
        return logFailure();

    // open first usable device
    VRmDeviceKey* p_device_key = 0;
    for (VRmDWORD i = 0; i < size && !device; ++i) {
        if (!VRmUsbCamGetDeviceKeyListEntry(i, &p_device_key))
            return logFailure();
        if (!p_device_key->m_busy) {
            if (!VRmUsbCamOpenDevice(p_device_key, &device))
                return logFailure();
        }
        if (!VRmUsbCamFreeDeviceKey(&p_device_key))
            return logFailure();
    }

    // display error when no camera has been found
    if (!device) {
        std::cerr << "No suitable VRmagic device found!" << std::endl;
        return logFailure();
    }

    // NOTE:
    // from now on, the "device" handle can be used to access the camera board.
    // use VRmUsbCamCloseDevice to end the usage

    if (!VRmUsbCamResetFrameCounter(device))
        return logFailure();

    // start grabber at first
    if (!VRmUsbCamStart(device))
        return logFailure();

    return opened = true;
}

bool VRmUsbCam::read(cv::Mat& frame)
{
    // lock next (raw) image for read access, convert it to the desired
    // format and unlock it again, so that grabbing can
    // go on
    VRmImage* p_source_img = 0;
    VRmDWORD frames_dropped = 0;
    if(!VRmUsbCamLockNextImageEx(device, port, &p_source_img, &frames_dropped))
    {
        // in case of an error, check for trigger timeouts and trigger stalls.
        // both can be recovered, so continue. otherwise exit the app
        if(VRmUsbCamLastErrorWasTriggerTimeout())
            std::cerr << "trigger timeout" << std::endl;
        else if(VRmUsbCamLastErrorWasTriggerStall())
            std::cerr << "trigger stall" << std::endl;
        else
            return logFailure();
    }

    // note: p_source_img may be null in case a recoverable error
    // (like a trigger timeout) occured.
    // in this case, we just pump GUI events and then continue with the loop
    if (p_source_img) {
        VRmDWORD frame_counter;
        if(!VRmUsbCamGetFrameCounter(p_source_img,&frame_counter))
            return logFailure();

        // see, if we had to drop some frames due to data transfer stalls. if so,
        // output a message
        if (frames_dropped)
            std::cout << "- " << frames_dropped <<  " frame(s) dropped -"
                      << std::endl;

        // get source image info
        VRmDWORD width = p_source_img->m_image_format.m_width;
        VRmDWORD height = p_source_img->m_image_format.m_height;
        VRmDWORD pixeldepth;
        VRmDWORD pitch = p_source_img->m_pitch;
        VRmUsbCamGetPixelDepthFromColorFormat(
                    p_source_img->m_image_format.m_color_format, &pixeldepth);

        // build cv::Mat from source image
        cv::Size frameSize(width, height);
        if (frame.empty() || frame.type() != CV_8U || frame.size() != frameSize)
            frame = cv::Mat(frameSize, CV_8U);

        // flip horizontal
        for (VRmDWORD x = 0, x2 = width-1; x < width; x++, x2--) {
            for (VRmDWORD y = 0, y2 = height-1; y < height; y++, y2--) {
                frame.data[(x*pixeldepth+y*width)] =
                        p_source_img->mp_buffer[(x2*pixeldepth+y2*pitch)];
            }
        }

        if(!VRmUsbCamUnlockNextImage(device,&p_source_img))
            return logFailure();

        // free the resources of the source image
        if(!VRmUsbCamFreeImage(&p_source_img))
            return logFailure();
    }

    return true;
}

void VRmUsbCam::release()
{
    opened = false;

    if(!VRmUsbCamStop(device))
        logFailure();

    if (!VRmUsbCamCloseDevice(device))
        logFailure();
}

bool VRmUsbCam::isOpened() const
{
    return opened;
}

cv::VideoCapture &VRmUsbCam::operator >>(cv::Mat& frame)
{
    read(frame);
    return *this;
}

void VRmUsbCam::initialize()
{
    if (initialized)
        return;

    // at first, be sure to call VRmUsbCamCleanup() at exit, even in case
    // of an error
    atexit(VRmUsbCamCleanup);

    // ...or on segmentation fault
    signal(SIGSEGV, forceShutdown);
    signal(SIGABRT, forceShutdown);

    initialized = true;
}

void VRmUsbCam::forceShutdown(int sig)
{
    signal(SIGSEGV, SIG_DFL);
    signal(SIGABRT, SIG_DFL);
    std::cerr << "Segfault, shutting down camera" << std::endl;
    VRmUsbCamCleanup();
}

bool VRmUsbCam::logFailure()
{
    std::cerr << "VRmUsbCam Error: " << VRmUsbCamGetLastError() << std::endl;
    return false;
}

bool VRmUsbCam::initialized = false;
