#include "Selector.h"

#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "Constants.h"

Selector::Selector(const std::string window, const cv::Mat& frame) :
    selection_valid(false),
    selecting(false),
    window(window),
    bounds(0, 0, frame.cols, frame.rows)
{
    frame.copyTo(this->frame);
    cv::setMouseCallback(window, mouse_callback, this);
}

Selector::~Selector()
{
    cv::setMouseCallback(window, nullptr, nullptr);
}

bool Selector::is_valid() const
{
    return selection_valid;
}

bool Selector::is_selecting() const
{
    return selecting;
}

cv::Rect Selector::get_selection() const
{
    return selection;
}

void Selector::mouse_callback(int event, int x, int y, int flags, void* data)
{
    Selector* self = (Selector*) data;

    switch(event)
    {
        // new selection
        case CV_EVENT_LBUTTONDOWN:
            self->selection_valid = false;
            self->selecting = true;
            self->selection = cv::Rect(0, 0, 0, 0);
            self->origin.x = x;
            self->origin.y = y;
            break;

        // selection finished
        case CV_EVENT_LBUTTONUP:
            if (self->selection.size() != cv::Size(0, 0) || // click
                self->selection.x < 0 || self->selection.y < 0) // out of window
                self->selection_valid = true;
            self->selecting = false;

        // on each event secept button-down (no break in button-up):
        default:
            if(self->selecting) // during selection, update rectangle
            {
                self->selection.x = std::min(x, self->origin.x);
                self->selection.y = std::min(y, self->origin.y);
                self->selection.width = std::abs(x - self->origin.x);
                self->selection.height = std::abs(y - self->origin.y);
                self->selection &= self->bounds;
            }

            // display selection
            cv::Mat selection_frame;
            self->frame.copyTo(selection_frame);
            cv::rectangle(selection_frame, self->selection, WHITE);
            cv::imshow(self->window, selection_frame);
    }
}
