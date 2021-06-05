# RealSense Depth Compression Algorithm

## Overview
This project deals with 16-bit dpeth image compression for RealseCameras. 
The idea is to convert the image to a seudo-vectorized shape, using curve fiting strategies.
This also has integration with OpenCV. 
The APP with both a sample file and also it  will open an OpenCV UI window and render colorized depth stream to it. 

For lossless compression, the ZSTD library is used.
The following code snippet is used to create `cv::Mat` from `rs2::frame`:
```cpp
// Query frame size (width and height)
const int w = depth.as<rs2::video_frame>().get_width();
const int h = depth.as<rs2::video_frame>().get_height();

// Create OpenCV matrix of size (w,h) from the colorized depth data
Mat image(Size(w, h), CV_8UC3, (void*)depth.get_data(), Mat::AUTO_STEP);
``` 
