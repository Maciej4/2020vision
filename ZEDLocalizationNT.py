print("Program start")

import time
import pyzed.sl as sl
from networktables import NetworkTables
import threading
import math
import sys

print("Imports Successful")

# ----- Networktables initialization -----

cond = threading.Condition()
notified = [False]

def connectionListener(connected, info):
    print(info, '; Connected=%s' % connected)
    with cond:
        notified[0] = True

        cond.notify()

NetworkTables.initialize(server='10.9.72.2')
#NetworkTables.initialize(server='192.168.86.254')
NetworkTables.addConnectionListener(connectionListener, immediateNotify=True)

with cond:
    print("Waiting")
    if not notified[0]:
        cond.wait()
        #print("Closing camera")
        #zed.close()
        #sys.exit()

print("Connected!")

table = NetworkTables.getTable('jetson')

table.putNumber('heartbeat', -1)

time.sleep(0.5)


# ----- ZED initilization -----

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

print("Camera init sucess")

table.putNumber('heartbeat', -2)

# ----- Tracking Loop -----

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
        #transform = "Translation: Tx: {0:+5.3f}, Ty: {1:+5.3f}, Tz {2:+5.3f}\n".format(tx, ty, tz)
        transform = "Translation: Tx: {0:+5.3f}, Ty: {1:+5.3f}, Tz {2:+5.3f}, Timestamp: {3:+}\n".format(tx, ty, tz, zed_pose.timestamp)

        py_orientation = sl.Orientation()
        rx = round(zed_pose.get_orientation(py_orientation).get()[0], 3)
        ry = round(zed_pose.get_orientation(py_orientation).get()[1], 3)
        rz = round(zed_pose.get_orientation(py_orientation).get()[2], 3)
        rw = round(zed_pose.get_orientation(py_orientation).get()[3], 3)

        print(transform)
        #rx, ry, rz = quaternion_to_euler(ox, oy, oz, ow)
        #print("Rx: {0:+5.3f}, Ry: {1:+5.3f}, Rz: {2:+5.3f}\n".format(rx, ry, rz))

        table.putNumber('zedTx', tx)
        table.putNumber('zedTy', ty)
        table.putNumber('zedTz', tz)
        table.putNumber('zedRx', rx)
        table.putNumber('zedRy', ry)
        table.putNumber('zedRz', rz)
        table.putNumber('zedRw', rw)
        #table.putNumber('heartbeat', zed_pose.timestamp)

#def quaternion_to_euler(x, y, z, w):
#    t0 = +2.0*(w*x+y*z)
#    t1 = +1.0-2.0*(x*x+y*y)
#    X = math.degrees(math.atan2(t0, t1))
#    
#    t2 = +2.0*(w*y-z*x)
#    t2 = +1.0 if t2 > +1.0 else t2
#    t2 = -1.0 if t2 < -1.0 else t2
#    Y = math.degrees(math.asin(t2))
#
#    t3 = +2.0*(w*z+x*y)
#    t4 = +1.0-2.0*(y*y+z*z)
#    Z = math.degrees(math.atan2(t3, t4))
#    
#    return X,Y,Z

if __name__ == "__main__":
    startTime = time.time()
    while True:
        track()
        table.putNumber('heartbeat', time.time() - startTime)
        time.sleep(.05)

# ----- Close camera and kill program -----
print("exiting")
zed.close()
sys.exit()
