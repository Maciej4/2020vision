import pyzed.sl as sl
import zmq
import threading
import time

killComms = False

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

transform = ''
# Create a Camera object
zed = sl.Camera()

# Create a InitParameters object and set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.RESOLUTION_HD720  # Use HD720 video mode (default fps: 60)
# Use a right-handed Y-up coordinate system
init_params.coordinate_system = sl.COORDINATE_SYSTEM.COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP
init_params.coordinate_units = sl.UNIT.UNIT_METER  # Set units in meters

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(1)

# Enable positional tracking with default parameters
py_transform = sl.Transform()  # First create a Transform object for TrackingParameters object
tracking_parameters = sl.TrackingParameters(init_pos=py_transform)
err = zed.enable_tracking(tracking_parameters)
if err != sl.ERROR_CODE.SUCCESS:
    exit(1)

# Track the camera position during 1000 frames
i = 0
zed_pose = sl.Pose()
zed_imu = sl.IMUData()
runtime_parameters = sl.RuntimeParameters()

def dataLoop():
    while True:
        print("Awaiting message...")
        message = socket.recv()
        print("Received request: %s" % message)
        time.sleep(1)
        socket.send(b"World")

def track():
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # Get the pose of the left eye of the camera with reference to the world frame
        zed.get_position(zed_pose, sl.REFERENCE_FRAME.REFERENCE_FRAME_WORLD)
        zed.get_imu_data(zed_imu, sl.TIME_REFERENCE.TIME_REFERENCE_IMAGE)

        # Display the translation and timestamp
        py_translation = sl.Translation()
        tx = round(zed_pose.get_translation(py_translation).get()[0], 3)
        ty = round(zed_pose.get_translation(py_translation).get()[1], 3)
        tz = round(zed_pose.get_translation(py_translation).get()[2], 3)
        transform = "Translation: Tx: {0:+5.3f}, Ty: {1:+5.3f}, Tz {2:+5.3f}\n".format(tx, ty, tz)
        #"Translation: Tx: {0:+5.3f}, Ty: {1:+5.3f}, Tz {2:+5.3f}, Timestamp: {3:+}\n".format(tx, ty, tz, zed_pose.timestamp)
        #print(transform)

        # Display the orientation quaternion
        '''py_orientation = sl.Orientation()
        ox = round(zed_pose.get_orientation(py_orientation).get()[0], 3)
        oy = round(zed_pose.get_orientation(py_orientation).get()[1], 3)
        oz = round(zed_pose.get_orientation(py_orientation).get()[2], 3)
        ow = round(zed_pose.get_orientation(py_orientation).get()[3], 3)
        rotation = "Orientation: Ox: {0:+5.3f}, Oy: {1:+5.3f}, Oz {2:+5.3f}, Ow: {3:+5.3f}\n".format(ox, oy, oz, ow)
        print(rotation)
        
        #Display the IMU acceleratoin
        acceleration = [0,0,0]
        zed_imu.get_linear_acceleration(acceleration)
        ax = round(acceleration[0], 3)
        ay = round(acceleration[1], 3)
        az = round(acceleration[2], 3)
        print("IMU Acceleration: Ax: {0}, Ay: {1}, Az {2}\n".format(ax, ay, az))

        #Display the IMU angular velocity
        a_velocity = [0,0,0]
        zed_imu.get_angular_velocity(a_velocity)
        vx = round(a_velocity[0], 3)
        vy = round(a_velocity[1], 3)
        vz = round(a_velocity[2], 3)
        print("IMU Angular Velocity: Vx: {0}, Vy: {1}, Vz {2}\n".format(vx, vy, vz))

        # Display the IMU orientation quaternion
        imu_orientation = sl.Orientation()
        ox = round(zed_imu.get_orientation(imu_orientation).get()[0], 3)
        oy = round(zed_imu.get_orientation(imu_orientation).get()[1], 3)
        oz = round(zed_imu.get_orientation(imu_orientation).get()[2], 3)
        ow = round(zed_imu.get_orientation(imu_orientation).get()[3], 3)
        print("IMU Orientation: Ox: {0}, Oy: {1}, Oz {2}, Ow: {3}\n".format(ox, oy, oz, ow))
        '''

if __name__ == "__main__":
    x = threading.Thread(target=dataLoop, daemon=True) #args=(1,), 
    x.start()

    print("Passed part")

    while True:
        track()
