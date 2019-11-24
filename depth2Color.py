## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
import time
import imutils
from imutils.video import FileVideoStream, WebcamVideoStream
from common import draw_str, draw_str_big
import argparse


# Create a pipeline
pipeline = rs.pipeline()

#Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)

# Start streaming
profile = pipeline.start(config)


parser = argparse.ArgumentParser(description='Depth 2 Color Magic')	
parser.add_argument("--source", "-s", required=True, default=0, help="Path to video file or integer representing webcam index"+ " (default 0).")
parser.add_argument("-o", "--output", required=False, help="path to output video")
args = parser.parse_args()


# initialize the video stream
print("[INFO] starting video stream...")
video_source = args.source

if video_source.isdigit():
	vs = WebcamVideoStream(src=int(video_source)).start()
else:
	vs = WebcamVideoStream(src=video_source).start()

time.sleep(2.0)
# setup video writer
writer = None
(W, H) = (None, None)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
		
        image = vs.read()
        dim = (1280, 720)
		
        # resize image
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) 

        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)

        alpha = 0.25
        cv2.addWeighted(bg_removed, alpha, image, 1 - alpha, 0, image)

        cv2.imshow('Align Example', image)
        output = image.copy()
        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args.output, fourcc, 30, (output.shape[1], output.shape[0]), True)
	
        if args.output:
            # write the output frame to disk
            writer.write(output)		
		
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()

# do a bit of cleanup
cv2.destroyAllWindows()
writer.release()
vs.stop()