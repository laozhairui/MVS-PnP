
#include <Camera.h>

#include <opencv2/imgproc.hpp>
int Camera::m_sdk_reference_count =0;

Camera::Camera()
    : m_device_list{}
    , m_handle(nullptr)
    , m_sdk_acquired(false)
    , m_device_opened(false)
    , m_grabbing(false)
    , m_last_error()
{
}

Camera::~Camera()
{
    finalize();
}

bool Camera::initialize()
{
    if(m_sdk_acquired)
    {
        return true;
    }
    m_last_error.clear();

    if(m_sdk_reference_count == 0)
    {
        int ret =MV_CC_Initialize();

        if(ret != MV_OK)
        {
            m_last_error = "Failed to initialize MVS SDK.";
            return false;
        }
    }

    ++m_sdk_reference_count;
    m_sdk_acquired = true;
    return true;
}

bool Camera::finalize()
{
    if (!m_sdk_acquired)
    {
        return true;
    }

    if(!close())
    {
        return false;
    }

    if (m_sdk_reference_count <= 0)
    {
        m_last_error = "Invaild SDK reference count.";
        return true;
    }


    if(m_sdk_reference_count == 1)
    {
        int ret = MV_CC_Finalize();

        if (ret != MV_OK)
        {
            m_last_error = "Failed to finalize MVS SDK.";
            return false;
        }
    }

    --m_sdk_reference_count;
    m_sdk_acquired = false;

    return true;
}

bool Camera::enumerate_devices()
{
    m_last_error.clear();
    m_device_list = {};

    if (!m_sdk_acquired)
    {
        m_last_error = "Initialize the MVS SDK first.";
        return false;
    }

    int ret = MV_CC_EnumDevices(
        MV_GIGE_DEVICE | MV_USB_DEVICE,
        &m_device_list
    );

    if (ret != MV_OK)
    {
        m_last_error = "Failed to enumerate camera devices.";
        return false;
    }

    if (m_device_list.nDeviceNum == 0)
    {
        m_last_error = "No camera device found.";
        return false;
    }

    return true;
}

int Camera::device_count() const
{
    return static_cast<int>(
        m_device_list.nDeviceNum
    );
}

bool Camera::create_handle(int device_id)
{
    m_last_error.clear();

    if (!m_sdk_acquired)
    {
        m_last_error = "Initialize the MVS SDK first.";
        return false;
    }

    if (m_handle != nullptr)
    {
        m_last_error = "Camera handle already exists.";
        return false;
    }

    if (device_id < 0
        || device_id >= static_cast<int>(m_device_list.nDeviceNum))
    {
        m_last_error = "Invalid camera device ID.";
        return false;
    }

    MV_CC_DEVICE_INFO* device_info =
        m_device_list.pDeviceInfo[device_id];

    if (device_info == nullptr)
    {
        m_last_error = "Camera device information is null.";
        return false;
    }

    int ret = MV_CC_CreateHandle(
        &m_handle,
        device_info
    );

    if (ret != MV_OK)
    {
        m_handle = nullptr;
        m_last_error = "Failed to create camera handle.";
        return false;
    }

    return true;
}

bool Camera::open()
{
    m_last_error.clear();

    if (!m_sdk_acquired)
    {
        m_last_error = "Initialize the MVS SDK first.";
        return false;
    }

    if (m_handle == nullptr)
    {
        m_last_error = "Create the camera handle before opening.";
        return false;
    }

    if (m_device_opened)
    {
        return true;
    }

    int ret = MV_CC_OpenDevice(m_handle);

    if (ret != MV_OK)
    {
        m_last_error = "Failed to open camera device.";
        std::string error_message = m_last_error;
        close();
        m_last_error = error_message;
        return false;
    }

    m_device_opened = true;

    if (!set_camera_parameters())
    {
        std::string error_message = m_last_error;

        close();

        m_last_error = error_message;

        return false;
    }

    return true;
}

bool Camera::set_camera_parameters()
{
    m_last_error.clear();

    if (!m_device_opened)
    {
        m_last_error = "Open camera before setting parameters.";
        return false;
    }

    int ret = MV_CC_SetEnumValue(
        m_handle,
        "ExposureAuto",
        0
    );

    if (ret != MV_OK)
    {
        m_last_error = "Failed to disable auto exposure.";
        return false;
    }

    ret = MV_CC_SetFloatValue(
        m_handle,
        "ExposureTime",
        20000.0F
    );

    if (ret != MV_OK)
    {
        m_last_error = "Failed to set exposure time.";
        return false;
    }

    ret = MV_CC_SetFloatValue(
        m_handle,
        "Gain",
        15.0F
    );

    if (ret != MV_OK)
    {
        m_last_error = "Failed to set gain.";
        return false;
    }

    return true;
}

bool Camera::start_grabbing()
{
    m_last_error.clear();

    if (!m_device_opened)
    {
        m_last_error =
            "Open the camera before starting acquisition.";
        return false;
    }

    if (m_grabbing)
    {
        return true;
    }

    int ret = MV_CC_StartGrabbing(m_handle);

    if (ret != MV_OK)
    {
        m_last_error = "Failed to start image acquisition.";
        return false;
    }

    m_grabbing = true;

    return true;
}

bool Camera::get_frame(cv::Mat& image)
{
    m_last_error.clear();

    if (!m_grabbing)
    {
        m_last_error =
            "Start image acquisition before getting frames.";
        return false;
    }

    MV_FRAME_OUT frame_info{};

    int ret = MV_CC_GetImageBuffer(
        m_handle,
        &frame_info,
        1000
    );

    if (ret != MV_OK)
    {
        m_last_error = "Failed to get image buffer.";
        return false;
    }

    bool convert_success = convert_to_bgr(
        frame_info,
        image
    );

    int free_ret = MV_CC_FreeImageBuffer(
        m_handle,
        &frame_info
    );

    if (!convert_success)
    {
        image.release();
        m_last_error = "Failed to convert image to BGR format.";
        return false;
    }

    if (free_ret != MV_OK)
    {
        image.release();
        m_last_error = "Failed to release image buffer.";
        return false;
    }

    return true;
}

bool Camera::convert_to_bgr(
    MV_FRAME_OUT& frame_info,
    cv::Mat& image
)
{
    if (frame_info.pBufAddr == nullptr)
    {
        return false;
    }

    int width = static_cast<int>(
        frame_info.stFrameInfo.nWidth
    );

    int height = static_cast<int>(
        frame_info.stFrameInfo.nHeight
    );

    if (width <= 0 || height <= 0)
    {
        return false;
    }

    cv::Mat raw_image(
        height,
        width,
        CV_8UC1,
        frame_info.pBufAddr
    );

    cv::cvtColor(
        raw_image,
        image,
        cv::COLOR_BayerBG2BGR
    );

    return !image.empty();
}
bool Camera::stop_grabbing()
{
    if (!m_grabbing)
    {
        return true;
    }

    int ret = MV_CC_StopGrabbing(m_handle);

    if (ret != MV_OK)
    {
        m_last_error = "Failed to stop image acquisition.";
        return false;
    }

    m_grabbing = false;

    return true;
}

bool Camera::close()
{
    if (m_handle == nullptr)
    {
        m_device_opened = false;
        m_grabbing = false;
        return true;
    }

    if (m_grabbing)
    {
        if (!stop_grabbing())
        {
            return false;
        }
    }

    if (m_device_opened)
    {
        int ret = MV_CC_CloseDevice(m_handle);

        if (ret != MV_OK && m_last_error.empty())
        {
            m_last_error = "Failed to close camera device.";
        }

        m_device_opened = false;
        m_grabbing = false;
    }

    int ret = MV_CC_DestroyHandle(m_handle);

    if (ret != MV_OK && m_last_error.empty())
    {
        m_last_error = "Failed to destroy camera handle.";
    }

    m_handle = nullptr;
    m_device_opened = false;
    m_grabbing = false;

    return true;
}

bool Camera::is_opened() const
{
    return m_device_opened;
}

std::string Camera::last_error() const
{
    return m_last_error;
}
