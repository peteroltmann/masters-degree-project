#ifndef VRM_UBS_H
#define VRM_UBS_H

#include <vrmusbcam2.h>
#include <opencv2/highgui/highgui.hpp>


/*!
 * \brief VRmagic USB camera accessing class.
 */
class VRmUsbCam : public cv::VideoCapture
{
public:
    VRmUsbCam(); //!< The default constructor.
    virtual ~VRmUsbCam(); //!< The default destructor.

    /*!
     * \brief Open VRmagic camera device for video capture.
     * \return true on success.
     */
    virtual bool open(int deviceNo);

    /*!
     * \brief Get next frame of the video capture.
     * \param frame the Mat where is frame is to be stored.
     */
    virtual bool read(cv::Mat& frame);

    /*!
     * \brief Release VRmagic camera device.
     */
    virtual void release();

    /*!
     * \brief Indicates if camera device is opened.
     * \return true if the camera device is opened.
     */
    virtual bool isOpened() const;

    /*!
     * \brief Get next frame of the video capture.
     * \param frame the Mat where is frame is to be stored
     * \return A reference to this object.
     */
    virtual VideoCapture& operator >> (cv::Mat& frame);

    virtual double get(int propId);

private:

    VRmUsbCamDevice device; //!< opened camera device.
    VRmDWORD port; //!< opened camera port.
    bool opened; //!< indicates if camera device is opened.

    // static members
    static void initialize(); //!< initialize once to cleanup at exit.
    static void forceShutdown(int sig); //!< react on segfault.
    static bool logFailure(); //!< print out last error and return false.

    static bool initialized; //!< indicates if cleanup at exit is intitialized.
};

#endif // VRM_UBS_H
