# -*- coding: utf-8 -*-
import pyzed.sl as sl
import numpy
from asciimatics.screen import Screen

import math
from collections import namedtuple

Quaternion = namedtuple('Quaternion', 'w x y z')
Euler = namedtuple('Euler', 'x y z')

def crop_rotation(angle):
    if angle > 180:
        return angle-360
    elif angle < -180:
        return angle+360
    else:
        return angle

def convert_to_radians(degrees):
    return (degrees/180) * math.pi

def ctd(radians):
    return (radians*180) / math.pi

def quaternion_to_euler(q):
    sqw = q.w * q.w
    sqx = q.x * q.x
    sqy = q.y * q.y
    sqz = q.z * q.z

    normal = math.sqrt(sqw + sqx + sqy + sqz)
    pole_result = (q.x * q.z) + (q.y * q.w)

    if (pole_result > (0.5 * normal)): # singularity at north pole
        ry = math.pi/2 #heading/yaw?
        rz = 0 #attitude/roll?
        rx = 2 * math.atan2(q.x, q.w) #bank/pitch?
        return Euler(rx, ry, rz)
    if (pole_result < (-0.5 * normal)): # singularity at south pole
        ry = -math.pi/2
        rz = 0
        rx = -2 * math.atan2(q.x, q.w)
        return Euler(rx, ry, rz)

    r11 = 2*(q.x*q.y + q.w*q.z)
    r12 = sqw + sqx - sqy - sqz
    r21 = -2*(q.x*q.z - q.w*q.y)
    r31 = 2*(q.y*q.z + q.w*q.x)
    r32 = sqw - sqx - sqy + sqz

    rx = math.atan2( r31, r32 )
    if -1 <= r21 <= 1:
        ry = math.asin ( r21 )
    else:
        ry = numpy.sign( r21 ) * 90
    rz = math.atan2( r11, r12 )

    return Euler(rx, ry, rz)
    
def euler_to_quaternion(angle_x, angle_y, angle_z):
    heading  = convert_to_radians(angle_y)
    attitude = convert_to_radians(angle_z)
    bank     = convert_to_radians(angle_x)

    c1 = math.cos(heading/2)
    c2 = math.cos(attitude/2)
    c3 = math.cos(bank/2)

    s1 = math.sin(heading/2)
    s2 = math.sin(attitude/2)
    s3 = math.sin(bank/2)

    w = (c1 * c2 * c3) - (s1 * s2 * s3)
    x = (s1 * s2 * c3) + (c1 * c2 * s3)
    y = (s1 * c2 * c3) + (c1 * s2 * s3)
    z = (c1 * s2 * c3) - (s1 * c2 * s3)
    
    return Quaternion(w, x, y, z)

def minecraft_inator(mesh):
    output_2d = numpy.empty((100, 100))
    output_2d.fill(0)
    sim_count = 0

    #output_2d = [[0]*100 for i in range(100)]
    #for f in range(100):
    #    output_2d.append([])
    #    for r in range(100):
    #        output_2d[f].append(0)

    for chunk in mesh.chunks:
        #print("chunk added")
        for vertex in chunk.vertices:
            #print("vertex questioned")
            if -0.2 < vertex[1] < 0.2 and sim_count % 10 == 0:
                output_2d[int(vertex[0] * 10)+50][int(vertex[2] * 10)+50] += 1
                #print("vertex added: {0}, {1}, {2}".format(int(vertex[0] * 10)+50, vertex[1], int(vertex[2] * 10)+50))
            sim_count += 1
    return output_2d

def main(screen):
    screen.print_at("Intializing Camera:", 1, 1)
    screen.refresh()

    # Create a Camera object
    zed = sl.Camera()

    screen.print_at("[    ]", 22, 1)
    screen.refresh()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.RESOLUTION_HD720  # Use HD720 video mode (default fps: 60)
    # Use a right-handed Y-up coordinate system
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP
    init_params.coordinate_units = sl.UNIT.UNIT_METER  # Set units in meters

    screen.print_at("[█   ]", 22, 1)
    screen.refresh()

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    screen.print_at("[██  ]", 22, 1)
    screen.refresh()

    # Enable positional tracking with default parameters
    # First create a Transform object for TrackingParameters object
    py_transform = sl.Transform()
    tracking_parameters = sl.TrackingParameters(init_pos=py_transform)
    err = zed.enable_tracking(tracking_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    screen.print_at("[███ ]", 22, 1)
    screen.refresh()

    # Enable spatial mapping
    mapping_parameters = sl.SpatialMappingParameters(map_type=sl.SPATIAL_MAP_TYPE.SPATIAL_MAP_TYPE_MESH)
    err = zed.enable_spatial_mapping(mapping_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    screen.print_at("[████]", 22, 1)
    screen.refresh()

    # Track the camera position during 1000 frames
    i = 0
    zed_pose = sl.Pose()
    zed_imu = sl.IMUData()
    py_fpc = sl.Mesh()  # Create a Mesh object
    runtime_parameters = sl.RuntimeParameters()
    final = []

    while i < 100000:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Get the pose of the left eye of the camera with reference to the world frame
            zed.get_position(zed_pose, sl.REFERENCE_FRAME.REFERENCE_FRAME_WORLD)
            zed.get_imu_data(zed_imu, sl.TIME_REFERENCE.TIME_REFERENCE_IMAGE)

            # In the background, spatial mapping will use newly retrieved images, depth and pose to update the mesh
            mapping_state = zed.get_spatial_mapping_state()

            # Display the translation and timestamp
            py_translation = sl.Translation()
            tx = round(zed_pose.get_translation(py_translation).get()[0], 3)
            ty = round(zed_pose.get_translation(py_translation).get()[1], 3)
            tz = round(zed_pose.get_translation(py_translation).get()[2], 3)
            screen.print_at("Translation: Tx: {0:+1.3f}, Ty: {1:+1.3f}, Tz {2:+1.3f}".format(tx, ty, tz), 1, 1)

            # Display the orientation quaternion
            py_orientation = sl.Orientation()
            ox = round(zed_pose.get_orientation(py_orientation).get()[0], 3)
            oy = round(zed_pose.get_orientation(py_orientation).get()[1], 3)
            oz = round(zed_pose.get_orientation(py_orientation).get()[2], 3)
            ow = round(zed_pose.get_orientation(py_orientation).get()[3], 3)
            q_all = Quaternion(ox, oy, oz, ow)
            e_all = quaternion_to_euler(q_all)
            screen.print_at("Orientation: Rx: {0:+1.3f}, Ry: {1:+1.3f}, Rz, {0:+1.3f}".format(ctd(e_all.x), ctd(e_all.y), ctd(e_all.z)), 1, 3)
            #screen.print_at("Orientation: Ox: {0:+1.3f}, Oy: {1:+1.3f}, Oz {2:+1.3f}, Ow: {3:+1.3f}".format(ox, oy, oz, ow), 1, 3)

            screen.print_at("Timestamp: {0}; Press q to quit".format(zed_pose.timestamp), 1, 5)

            robot_x = int(tx * 10) + 50
            robot_y = ty
            robot_z = int(tz * 10) + 50
            robot_r = ctd(e_all.y)

            robot_char = "🢁"

            #if -45 <= robor_r <= 45:

            screen.print_at("▓", robot_x, robot_z)# 🢀🢂🢁🢃🢄🢅🢆🢇

            # Request an update of the spatial map every 30 frames (0.5s in HD720 mode)
            if i % 30 == 0 :
                zed.request_spatial_map_async()

            # Retrieve spatial_map when ready
            if zed.get_spatial_map_request_status_async() == sl.ERROR_CODE.SUCCESS and i > 0 :
                zed.retrieve_spatial_map_async(py_fpc)
                map = minecraft_inator(py_fpc)
                print(len(py_fpc.chunks))
                for x in range(100):
                    for y in range(100):
                        if map[x][y] > 0:
                            screen.print_at("█", x, y)
                #i = 1000

            ev = screen.get_key()
            if ev in (ord('Q'), ord('q')):
                i = 100000

            screen.refresh()

            screen.print_at(" ", robot_x, robot_z)

            i = i + 1

    # Extract, filter and save the mesh in an obj file
    print("Extracting Point Cloud...\n")
    err = zed.extract_whole_spatial_map(py_fpc)
    print(repr(err))
    ##print("Filtering Mesh...\n")
    ##py_mesh.filter(sl.MeshFilterParameters())  # Filter the mesh (remove unnecessary vertices and faces)
    #print("Saving Point Cloud...\n")
    #py_fpc.save("fpc.obj")

    #print(map)

    # Disable tracking and mapping and close the camera
    zed.disable_spatial_mapping()
    zed.disable_tracking()
    zed.close()

if __name__ == "__main__":
    Screen.wrapper(main)
