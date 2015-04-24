# Tracking Objects with Particle Filters and Active Contours

This software was developed during my master's degree.
It is capable of tracking the contours of deforming object in changing environments even if partial occlusions ocurr.

## Requirements

There are some requirements for this software:

* C++11
* CMake 2.8+
* OpenCV v2.9.11
* Boost v1.57.0

## Compilation

Follow the instructions listed below to compile the software.

* Run <code>cmake</code> from the <code>build</code>-directory 
    * <code>build</code>-directory free choosable, just not the source directory
    * Generate system-dependant Makefiles , e.g.:  
    <code>-G 'Unix Makefiles'</code>  
    <code>-G 'MinGW Makefiles'</code>
    * Specify build type:  
    <code>-DCMAKE_BUILD_TYPE=Debug</code>  
    <code>-DCMAKE_BUILD_TYPE=Release</code>
    * Set compiler flags (optional):  
    <code>-DSHOW_CONTOUR_EVOLUTION</code>  
    <code>-DTIME_MEASUREMENT</code>
* Run <code>make</code>

For example:

```
mkdir build && cd build
cmake -G 'Unix Makefiles' -DCMAKE_BUILD_TYPE=Release ..
make
./pf_hist_ac -h # print help information
```

## Parameterization

Parameterization is done by an YAML file, that can be specified via the command line arguments <code>-f \<filename\>.yml</code>.
Alternatively one can use the standard file <code>../parameterization.yml</code> (parent directory of the build directory).

An exemplary parameterization file incl. description of all parameters is listed below.

```
%YAML:1.0

# number of particles, default: 100
num_particles: 100

# number of iterations, default: 10
num_iterations: 150

# sigma for weight calculation, default: 20
sigma: 20

# histogram matching threshold, default: 0.25
bc_threshold: 0.25

# histogram matchimg threshold for adpation, default: 0.1, 0: no adaption
bc_threshold_adapt: 0.1

# histogram adaption factor (0 for using default)
a: 0.1

# number of fourier frequencies for filtering, default: 10
# the value used ist 2 * num_fourier to always obtain an even-numbered value
num_fourier:  7

# fourier descriptor matching threshold, default: 0.15, 1: none
fd_threshold: 0.15

# 1: select starting rectangle with mouse, 0: use specified
select_start_rect: 0

# starting rectangle for contour evolution on first frame
start_rect: [  35, 150,  40,  20 ] # fish

# path to input video or image sequence
input_path: "../input/palau2_gray_cropped/palau2_frames_%04d.png"

# number of input device, only used if input_path is empty, default: 0
input_device: 0

# charateristic views, list of strings to filepaths
char_views: ["../input/char_views/cv_fish/cv_fish_back_right_down_s.png",
             "../input/char_views/cv_fish/cv_fish_back_right_up_s.png"]

# file name for video and image output, leave empty to use 'out'
# file extension is added automatically
# output_file_name: "fish_result"

# path to output video direvtory, leave empty for no output
# output_path: "../output/fish/"

# frames per second for saved video, <= 0 or empty for taking source fps
fps: 25.0

# path to output directory for each frame as an image
# detailed view is added to 'details/' that needs to be created in advance
# save_img_path: "../output/fish/fish_img/"

# matching values that can be read in with matlab, leave empty for no output
# matlab_file_path: "../matlab/matlab_out.m"

# contour evolution detailed parameters, defaults used if not specified

# 0: CHAN_VESE (UM), 1: YEZZI (MS), default: 0
method: 0  

# true/false (0/1), default: 0               
localized: 0

# radius of localized regions, default: 18
rad: 18  

# curvature weight (higher -> smoother), default: 0.2
alpha: 0.2
```

## Usage

```
Usage:
  ./pf_hist_ac [-f <param-file>] [-c]

Options:
  -f <param-file>  Specify parameterization file, default: ../parameterization.yml
  -c               Use only single image contour evolution

Controls:
  Press 'q' to quit
  Press 'p' to show/hide particle filter estimate
  Press 'd' to toggle detailed output
  Press 't' to show/hide template
  Press 'space' pause/resume
```