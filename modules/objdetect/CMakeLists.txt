set(the_description "Object Detection")

option(OPENCV_ENABLE_YUNET_QR "Enable YUNET QR coarse detection" ON)

ocv_define_module(objdetect
    opencv_core
    opencv_imgproc
    opencv_3d
    OPTIONAL
        opencv_dnn  # Ensure opencv_dnn is included for DNN module support
    WRAP
        python
        java
        objc
        js
)

# ============================================================
#      YUNET QR DETECTOR USING OPENCV DNN (No ONNXRuntime)
# ============================================================

if(OPENCV_ENABLE_YUNET_QR)
    message(STATUS "YUNET QR: ON – using OpenCV DNN instead of ONNXRuntime")

    # Ensure OpenCV DNN module is linked
    target_link_libraries(opencv_objdetect PRIVATE opencv_dnn)

    # Message to indicate the change
    message(STATUS "Using OpenCV DNN for YUNET QR detection")
endif()

# ============================================================
#                   END YUNET QR CONFIG
# ============================================================
