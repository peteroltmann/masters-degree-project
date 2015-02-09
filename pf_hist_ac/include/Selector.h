#ifndef SELECTOR_H
#define SELECTOR_H

#include <opencv2/core/core.hpp>

/*!
 * \brief Selector to handle rectangular selection from a window and frame.
 */
class Selector
{
private:
    bool selection_valid;   //!< valid flag
    bool selecting;         //!< set during selection (mouse button down)
    cv::Rect selection;     //!< the selection rectanble
    cv::Point origin;       //!< selection origin (point of mouse button down)
    cv::string window;      //!< name of observed window
    cv::Mat frame;          //!< the frame used for selection
    cv::Rect bounds;        //!< frame bounds [0, 0, frame.cols, frame.rols]

public:
    /*!
     * \brief Construct a Selector.
     * \param window    name of observed window. Has to be created with
     *                  <tt>cv::namedWindow()</tt>. Unless, the mouse callback
     *                  can not be set.
     * \param frame     the frame used for selection
     */
    Selector(const std::string window, const cv::Mat& frame);

    virtual ~Selector(); //!< The default destructor
    bool is_valid() const; //!< \return weather the selection is valid or not
    bool is_selecting() const; //!< \return weather the selection is created
    cv::Rect get_selection() const; //!< \return the selection

private:
    //!< Mouse callback for an OpenCV window.
    static void mouse_callback(int event, int x, int y, int flags, void* data);
};

#endif // SELECTOR_H
