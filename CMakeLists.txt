cmake_minimum_required(VERSION 2.8)
#Name your project here
project(beleg1)
find_package( OpenCV REQUIRED )

#Sends the -std=c99 flag to the gcc compiler
#add_definitions(-std=c99)
add_definitions(-std=c++0x)
#add_definitions(-std=c++11)
#This tells CMake to fib.c and name it fibonacci
add_executable(beleg1 src/main.cpp)
set(OpenCV_LIBS 
    /usr/local/lib/libopencv_calib3d.so
    /usr/local/lib/libopencv_contrib.so
    /usr/local/lib/libopencv_core.so
    /usr/local/lib/libopencv_features2d.so
    /usr/local/lib/libopencv_flann.so
    /usr/local/lib/libopencv_gpu.so
    /usr/local/lib/libopencv_highgui.so
    /usr/local/lib/libopencv_imgproc.so
    /usr/local/lib/libopencv_legacy.so
    /usr/local/lib/libopencv_ml.so
    /usr/local/lib/libopencv_nonfree.so
    /usr/local/lib/libopencv_objdetect.so
    /usr/local/lib/libopencv_photo.so
    /usr/local/lib/libopencv_stitching.so
    /usr/local/lib/libopencv_superres.so
    /usr/local/lib/libopencv_ts.so
    /usr/local/lib/libopencv_video.so
    /usr/local/lib/libopencv_videostab.so
)
target_link_libraries( beleg1 ${OpenCV_LIBS} )

