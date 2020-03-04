import time
import threading
import cv2
import context
import datetime

try:
    from greenlet import getcurrent as get_ident
    print("using greenlet")
except ImportError:
    try:
        from thread import get_ident
        print("using thread")
    except ImportError:
        from _thread import get_ident
        print("using _thread")


def gstreamer_pipeline(
    capture_width=640,
    capture_height=480,
    display_width=640,
    display_height=480,
    framerate=120,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def generate_out():
    file_name = str(datetime.datetime.now().strftime("/home/pi/Documents/captures/%H:%M:%S-%d-%m-%y.mp4"))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    return cv2.VideoWriter(file_name, fourcc, 30.0, (640, 480))


video_len = 30
out = generate_out()


class CameraEvent(object):
    """An Event-like class that signals all active clients when a new frame is
    available.
    """
    def __init__(self):
        self.events = {}

    def wait(self):
        """Invoked from each client's thread to wait for the next frame."""
        ident = get_ident()
        if ident not in self.events:
            # this is a new client
            # add an entry for it in the self.events dict
            # each entry has two elements, a threading.Event() and a timestamp
            self.events[ident] = [threading.Event(), time.time()]
        return self.events[ident][0].wait()

    def set(self):
        """Invoked by the camera thread when a new frame is available."""
        now = time.time()
        remove = None
        for ident, event in self.events.items():
            if not event[0].isSet():
                # if this client's event is not set, then set it
                # also update the last set timestamp to now
                event[0].set()
                event[1] = now
            else:
                # if the client's event is already set, it means the client
                # did not process a previous frame
                # if the event stays set for more than 5 seconds, then assume
                # the client is gone and remove it
                if now - event[1] > 5:
                    remove = ident
        if remove:
            del self.events[remove]

    def clear(self):
        """Invoked from each client's thread after a frame was processed."""
        self.events[get_ident()][0].clear()


class Camera(object):
    thread = None  # background thread that reads frames from camera
    frame = None  # current frame is stored here by background thread
    last_access = 0  # time of last client access to the camera
    event = CameraEvent()

    def __init__(self):
        """Start the background camera thread if it isn't running yet."""
        if Camera.thread is None:
            Camera.last_access = time.time()

            # start background frame thread
            Camera.thread = threading.Thread(target=self._thread)
            context.threads.append(Camera.thread)
            Camera.thread.start()

            # wait until frames are available
            while context.keep_running:
                frame = self.get_frame()
                if frame is not None:
                    break
                time.sleep(0.1)

    def get_frame(self):
        """Return the current camera frame."""
        Camera.last_access = time.time()

        # wait for a signal from the camera thread
        Camera.event.wait()
        Camera.event.clear()

        return Camera.frame

    @staticmethod
    def frames():
        global out, video_len
        camera = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
        # camera = cv2.VideoCapture(0)
        # camera.set(3, 640)
        # camera.set(4, 480)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        ret, img = camera.read()
        if ret:
            print("image size: {0}".format(img.shape))
        else:
            print("Could not read from camera.")

        out_start = time.time()
        while context.keep_running:
            # read current frame
            _, img = camera.read()

            # start = time.time()
            out.write(img)
            # print("dt: {0}".format((1000*(time.time()-start))))

            if out_start + video_len < time.time():
                out_start = time.time()
                out = generate_out()

            # encode as a jpeg image and return both
            yield img

        print("\nReleasing Camera")
        out.release()
        camera.release()

    @classmethod
    def _thread(cls):
        """Camera background thread."""
        print('Starting camera thread.')
        frames_iterator = cls.frames()
        for frame in frames_iterator:
            Camera.frame = frame
            Camera.event.set()  # send signal to clients
            time.sleep(0)

            # if there hasn't been any clients asking for frames in
            # the last 10 seconds then stop the thread
            if time.time() - Camera.last_access > 10:
                frames_iterator.close()
                print('Stopping camera thread due to inactivity.')
                break
        Camera.thread = None
