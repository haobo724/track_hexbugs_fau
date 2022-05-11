# extract the OpenCV version info
import time

import cv2

# if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
# function to create our object tracker
# otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
# approrpiate object tracker constructor:
# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations

# OPENCV_OBJECT_TRACKERS = {
# 	"csrt": cv2.TrackerCSRT_create,
# 	"kcf": cv2.TrackerKCF_create,
# 	"boosting": cv2.TrackerBoosting_create,
# 	"mil": cv2.TrackerMIL_create,
# 	"tld": cv2.TrackerTLD_create,
# 	"medianflow": cv2.TrackerMedianFlow_create,
# 	"mosse": cv2.TrackerMOSSE_create
# }
# grab the appropriate object tracker using our dictionary of
# OpenCV object tracker objects
tracker_name='kcf'
# tracker = OPENCV_OBJECT_TRACKERS[tracker_name]()
tracker = cv2.TrackerCSRT_create()
# initialize the bounding box coordinates of the object we are going
# to track
# if a video path was not supplied, grab the reference to the web cam
video_path='OpenCV Object Tracking Demo #2.mkv'
vs = cv2.VideoCapture(video_path)
# initialize the FPS throughput estimator
# fps = None
# loop over frames from the video stream
ret, frame = vs.read()

cv2.imshow("Frame", frame)

initBB = cv2.selectROI("Frame", frame, fromCenter=False,
							   showCrosshair=True)
tracker.init(frame, initBB)

while True:
	# grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
	ret,frame = vs.read()
	if not ret :
		# print('something happened')
		continue
	# frame = frame[1] if args.get("video", False) else frame
	# check to see if we have reached the end of the stream

	if frame is None:
		break
	# resize the frame (so we can process it faster) and grab the
	# frame dimensions
	# frame = imutils.resize(frame, width=500)
	(H, W) = frame.shape[:2]
	# check to see if we are currently tracking an object
	if initBB is not None:
		# grab the new bounding box coordinates of the object
		(success, box) = tracker.update(frame)
		# check to see if the tracking was a success
		if success:
			(x, y, w, h) = [int(v) for v in box]
			cv2.rectangle(frame, (x, y), (x + w, y + h),
				(0, 255, 0), 2)
		# update the FPS counter
		# fps.update()
		# fps.stop()
		# initialize the set of information we'll be displaying on
		# the frame
		info = [
			("Tracker",tracker_name),
			("Success", "Yes" if success else "No"),
		]
		# loop over the info tuples and draw them on our frame
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
	# # show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(100) & 0xFF
		# if the 's' key is selected, we are going to "select" a bounding
		# box to track
	if key == ord("s"):
		# select the bounding box of the object we want to track (make
		# sure you press ENTER or SPACE after selecting the ROI)
		initBB = cv2.selectROI("Frame", frame, fromCenter=False,
							   showCrosshair=True)
		# start OpenCV object tracker using the supplied bounding box
		# coordinates, then start the FPS throughput estimator as well
		tracker.init(frame, initBB)
		# fps = FPS().start()
		# if the `q` key was pressed, break from the loop
	elif key == ord("q"):
		break
	# if we are using a webcam, release the pointer
vs.release()
# close all windows
cv2.destroyAllWindows()