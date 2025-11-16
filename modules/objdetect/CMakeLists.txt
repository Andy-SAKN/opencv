set(the_description "Object Detection")

option(OPENCV_ENABLE_YUNET_QR "Enable YUNET QR coarse detection" ON)

ocv_define_module(objdetect
    opencv_core
    opencv_imgproc
    opencv_3d
    OPTIONAL
        opencv_dnn
    WRAP
        python
        java
        objc
        js
)

# ============================================================
#      ONNXRUNTIME SUPPORT FOR CUSTOM YUNET QR DETECTOR
# ============================================================

if(OPENCV_ENABLE_YUNET_QR)
    message(STATUS "YUNET QR: ON – linking ONNXRuntime")

    # Homebrew installation path of ONNXRuntime
    set(ONNXRUNTIME_ROOT "/opt/homebrew/opt/onnxruntime")

    # Add ONNXRuntime headers
    target_include_directories(opencv_objdetect PRIVATE
        ${ONNXRUNTIME_ROOT}/include
    )

    # Add ONNXRuntime library directory
    target_link_directories(opencv_objdetect PRIVATE
        ${ONNXRUNTIME_ROOT}/lib
    )

    # Link the library
    target_link_libraries(opencv_objdetect PRIVATE
        onnxruntime
    )

    message(STATUS "Using ONNXRuntime include dir: ${ONNXRUNTIME_ROOT}/include")
    message(STATUS "Using ONNXRuntime lib dir:     ${ONNXRUNTIME_ROOT}/lib")
endif()

# ============================================================
#                   END YUNET QR CONFIG
# ============================================================
