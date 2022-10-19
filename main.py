'''
run the file ui.py to start the program.

if you want to use other video to test, please change the parameters such as maxCorners or NUM_OF_FEATURES
for the video that contains lots of objects, please increase the two parameters (usually 200-300 is regarded as good number)
'''

import cv2
import numpy as np

# Parameters for lucas kanade optical flow
lk_params = dict(winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# parameters for goodFeaturesToTrack
feature_params = dict( maxCorners = 50, qualityLevel = 0.3, minDistance = 7, blockSize = 7 )

NUM_OF_FEATURES = 50
COLOR_1 = (0,255,0)
COLOR_2 = (0,0,255)
DETECT_INTERVAL = 5
TRACKING_LEN = 100
TRACK_POINT = False
track_points_prev = []

# mode 2, with user interactions
def run_mode_2(filepath):
    global TRACK_POINT
    global track_points_prev

    cap = cv2.VideoCapture(filepath)

    track_points_prev = np.asarray(track_points_prev, dtype=np.float32).reshape(-1, 1, 2)

    HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('mode_2_result.mp4', fourcc, fps, (int(WIDTH), int(HEIGHT)))

    first_ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # capture the rectangle drawn by user
    x_rec, y_rec, w_rec, h_rec = draw_rectangle(prev_frame)

    # capture mouse events
    cv2.namedWindow('Optical Flow')
    cv2.setMouseCallback('Optical Flow', draw_point)

    first_gray = prev_gray[y_rec:y_rec+h_rec, x_rec:x_rec+w_rec]
    prev_p = cv2.goodFeaturesToTrack(first_gray, mask=None, **feature_params)
    for i in range(len(prev_p)):
        prev_p[i][0][0] += x_rec
        prev_p[i][0][1] += y_rec


    tracks = []
    for x, y in np.float32(prev_p).reshape(-1, 2):
        tracks.append([(x, y)])

    frame_index = 1

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = frame.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # update the tracking points (selected by user)
        if TRACK_POINT:
            frame = update_track_point(prev_gray, frame_gray, frame)

        # calculate the optical flow of good feature points
        new_p, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_p, None, **lk_params)

        if new_p is None:
            # print("No tracking points.")
            cv2.imshow("Optical Flow", frame)
            prev_gray = frame_gray.copy()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                track_points_prev = []
                TRACK_POINT = False
                break
            continue

        status = status.reshape(-1)

        k = 0
        for i, (new, old, st) in enumerate(zip(new_p, prev_p, status)):
            if (st == 0):
                continue
            a, b = new.ravel()
            c, d = old.ravel()
            dist = cal_dist(a,b,c,d)

            # delete the static points
            if dist>0.2:
                new_p[k] = new_p[i]
                tracks[k] = tracks[i]
                k +=1

        # extract the dynamic points
        new_p = new_p[:k]
        tracks = tracks[:k]

        # append the current dynamic point into the tracks[], if the tracking length larger than the TRACKING_LEN, delet the line
        tracks = append_to_tracks(tracks, new_p)

        # draw the line
        for tr in tracks:
            cv2.polylines(frame, [np.int32(tr)], False, COLOR_1, 1)

        # if the point out of the boundary, delete the lines
        tracks = del_out_of_boundary(tracks, WIDTH, HEIGHT)

        cv2.imshow('Optical Flow', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            track_points_prev = []
            TRACK_POINT = False
            break

        # update
        prev_gray = frame_gray.copy()
        prev_p = np.float32([tr[-1] for tr in tracks]).reshape(-1,1,2)
        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

# mode 1, original mode
def run_mode_1(filepath):

    cap = cv2.VideoCapture(filepath)
    cv2.namedWindow('Optical Flow')
    cv2.setMouseCallback('Optical Flow', draw_point)

    HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('mode_1_result.mp4', fourcc, 30.0, (int(WIDTH), int(HEIGHT)))

    # read the first frame and get the good feature points.
    prev_p, prev_gray, prev_frame = read_first_frame(cap)
    tracks = []
    for x, y in np.float32(prev_p).reshape(-1, 2):
        tracks.append([(x, y)])

    frame_index = 1
     # background subtractor
    bg_substr = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=36, detectShadows=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = frame.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # background subtract
        frame_bgsub = bg_substr.apply(frame_gray)
        frame_bgsub = cv2.medianBlur(frame_bgsub,5)
        frame_bgsub = cv2.medianBlur(frame_bgsub, 5)

        # find the contours
        ret, thresh = cv2.threshold(frame_bgsub, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame = draw_contours(contours, frame)

        # tracking the good points
        new_p, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_p, None, **lk_params)
        if new_p is None:
            cv2.imshow("Optical Flow", frame)
            prev_gray = frame_gray.copy()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        status = status.reshape(-1)

        k = 0
        for i, (new, old, st) in enumerate(zip(new_p, prev_p, status)):
            if (st == 0):
                continue
            a, b = new.ravel()
            c, d = old.ravel()
            dist = cal_dist(a, b, c, d)

            # delete the static points
            if dist > 0.2:
                new_p[k] = new_p[i]
                tracks[k] = tracks[i]
                k += 1

        # extract the dynamic points
        new_p = new_p[:k]
        tracks = tracks[:k]

        # append the current dynamic point into the tracks[], if the tracking length larger than the TRACKING_LEN, delet the line
        tracks = append_to_tracks(tracks, new_p)

        # draw the line
        for tr in tracks:
            cv2.polylines(frame, [np.int32(tr)], False, COLOR_1, 1)

        # re-detect corners
        if frame_index % DETECT_INTERVAL == 0:
            update_p = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)

            for p in update_p:
                if p in new_p.reshape(-1,1,2):
                    continue
                tracks.append([(p[0][0], p[0][1])])

        # if the point out of the boundary, delete the lines
        tracks = del_out_of_boundary(tracks, WIDTH, HEIGHT)

        cv2.imshow('Optical Flow', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # update
        prev_gray = frame_gray.copy()
        # prev_p = kp_new.reshape(-1, 1, 2)
        prev_p = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

# drawe the rectangle for the contours of the objects
def draw_contours(contours, frame):
    boundings = [cv2.boundingRect(cont) for cont in contours]
    for b in boundings:
        [x,y,w,h] = b
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), COLOR_2, 2)
    return frame

# update the coordinates of the tracking feature points
def update_track_point(prev_gray, frame_gray, frame):

    global track_points_prev
    track_points_prev = track_points_prev.reshape(-1, 1, 2)
    track_points_new, st, errors = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, track_points_prev, None, **lk_params)
    if track_points_new is None:
        return frame

    track_points_new = track_points_new[st == 1]

    track_points_new = track_points_new.reshape(-1, 2)
    for a, b in track_points_new:
        frame = cv2.circle(frame, (int(a), int(b)), 3, (0, 0, 255), 2)

    track_points_prev = track_points_new.reshape(-1, 1, 2)
    return frame

# delete the trajectories if the points out of the boundary
def del_out_of_boundary(tracks, width, height):
    new_tracks = []
    k = 0
    for i, kp in enumerate(np.float32([tr[-1] for tr in tracks])):
        x, y = kp.ravel()
        if (x < 0) or (y < 0) or (x > width) or (y > height):
            continue
        new_tracks.append(tracks[i])
        k += 1
    return new_tracks

# update array that store all the track points
def append_to_tracks(tracks, new_p):
    for i, kp in enumerate(new_p):
        x, y = kp.ravel()
        tracks[i].append((x, y))
        if len(tracks[i]) > TRACKING_LEN:
            del tracks[i][0]

    return tracks

# allow user to draw the rectangle (tracking area) and return the information of the area
def draw_rectangle(frame):
    roi = cv2.selectROI(windowName="Optical Flow", img=frame, showCrosshair=False, fromCenter=False)
    cv2.imshow('Optical Flow', frame)
    return roi[0], roi[1], roi[2], roi[3]

# capture mouse events and draw circle when user click, then add the point to the tracking array
def draw_point(event, x, y, flags, param):
    global TRACK_POINT
    global track_points_prev
    if event == cv2.EVENT_LBUTTONDOWN:
        x = np.float32(x)
        y = np.float32(y)
        track_points_prev = np.append(track_points_prev, [x, y])
        TRACK_POINT = True

def cal_dist(a,b,c,d):
    dist = (a-c)*(a-c)+(b-d)*(b-d)
    return dist

# use sift to detect the good feature points
def feature_detect(frame_gray, frame):
    sift = cv2.SIFT_create(nfeatures=NUM_OF_FEATURES)
    kp = sift.detect(frame_gray, None)

    kp_len = len(kp)
    pt = np.zeros((kp_len, 1, 2), dtype='float32')

    for i in range(kp_len):
        pt[i] = kp[i].pt
    # kp = np.float32([p for p in kp]).reshape(-1,1,2)
    cv2.drawKeypoints(frame_gray, kp, frame)

    # pt = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)

    # cv2.imshow('image', frame)

    return pt

# read the first frame and get the original good feature poitns
def read_first_frame(cap):

    first_ret, first_frame = cap.read()
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    pt = feature_detect(first_gray, first_frame)

    return pt, first_gray, first_frame

# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     PATH = "./video/test_9.mp4"
#     run_mode_1(PATH)
#     # run_mode_2(PATH)

