# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2

def main() :

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
 
    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    # start the video stream thread
    vs = VideoStream(src=0).start()
    fileStream = False

   
    time.sleep(1.0)
    
    # loop over frames from the video stream
    while True:
    	# if this is a file video stream, then we need to check if
    	# there any more frames left in the buffer to process
    	if fileStream and not vs.more():
    		break
    	# grab the frame from the threaded video file stream, resize
    	# it, and convert it to grayscale
    	# channels)
    	frame = vs.read()
    	frame = imutils.resize(frame, width=450)
    	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    	# detect faces in the grayscale frame
    	rects = detector(gray, 0)
    
    	# loop over the face detections
    	for rect in rects:
    		# determine the facial landmarks for the face region, then
    		# convert the facial landmark (x, y)-coordinates to a NumPy
    		# array
    		shape = predictor(gray, rect)
    		shape = face_utils.shape_to_np(shape)
            
    		# extract the left and right eye coordinates, then use the
    		# coordinates to compute the eye aspect ratio for both eyes
    		rightEye = shape[rStart:rEnd]        
    
    		# compute the convex hull for the left and right eye, then
    		# visualize each of the eyes
            
    		rightEyeHull = cv2.convexHull(rightEye)
    		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

     
    	# show the frame
    	cv2.imshow("Frame", frame)
    	key = cv2.waitKey(1) & 0xFF
     
    	# if the `q` key was pressed, break from the loop
    	if key == ord("q"):
    		break
    
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
if __name__ == '__main__' :
    main()