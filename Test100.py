import pyzed.sl as sl

print("Starting")

# Create a ZED camera object
zed = sl.Camera()

print("Camera Initialized")

# Set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.RESOLUTION_HD1080
init_params.camera_fps = 30

print("1")

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(-1)

print("2")

# Set the streaming parameters
stream = sl.StreamingParameters()
stream.codec = sl.STREAMING_CODEC.STREAMING_CODEC_AVCHD # can be AVCHD or HEVC
stream.bitrate = 8000
stream.port = 3000 # port used for sending the stream
# Enable streaming with the streaming parameters
status = zed.enable_streaming(stream)

print("3")

x = sl.RuntimeParameters(
    sl.SENSING_MODE.SENSING_MODE_STANDARD,
    True,
    True,
    sl.REFERENCE_FRAME.REFERENCE_FRAME_CAMERA
)

print(zed.grab(x) == sl.ERROR_CODE.SUCCESS)

try:
    while True :
        zed.grab(x)
except KeyboardInterrupt:
    pass

print("Ending")

# Disable streaming
zed.disable_streaming()
