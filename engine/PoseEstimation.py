import cv2
import mediapipe as mp
import pafy
import time
import numpy as np
import urllib.parse as urlparse
import imageio


class poseDetector:
    def __init__(
        self,
        mode=False,
        complex=1,
        smooth_landmarks=True,
        segmentation=True,
        smooth_segmentation=True,
        detectionCon=0.5,
        trackCon=0.5,
    ):

        self.mode = mode
        self.complex = complex
        self.smooth_landmarks = smooth_landmarks
        self.segmentation = segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpDrawStyle = mp.solutions.drawing_styles
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            self.mode,
            self.complex,
            self.smooth_landmarks,
            self.segmentation,
            self.smooth_segmentation,
            self.detectionCon,
            self.trackCon,
        )

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(
                    img,
                    self.results.pose_landmarks,
                    self.mpPose.POSE_CONNECTIONS,
                    # self.mpDrawStyle.get_default_pose_landmarks_style())
                    self.mpDraw.DrawingSpec(
                        color=(0, 0, 255), thickness=2, circle_radius=2
                    ),
                    self.mpDraw.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=2
                    ),
                )
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                x, y, z = lm.x, lm.y, lm.z
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):

        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the Angle
        radians = np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2)
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        # print(int(angle))

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), 2)
            cv2.putText(
                img,
                str(int(angle)) + "",
                (x2 - 50, y2 + 50),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 255, 255),
                2,
            )
        return angle


def main():
    url = input("Enter Youtube Video URL: ")
    url_data = urlparse.urlparse(url)
    query = urlparse.parse_qs(url_data.query)
    if url_data.path == "/watch":
        id = query["v"][0]
    elif url_data.path[:6] == "/embed":
        id = url_data.path[7:]
    else:
        id = url_data.path[1:]
    video = "https://youtu.be/{}".format(str(id))
    urlPafy = pafy.new(video)
    videoplay = urlPafy.getbest(preftype="any")
    cap = cv2.VideoCapture(videoplay.url)
    # cap = cv2.VideoCapture(0)
    milliseconds = 1000
    start_time = int(input("Enter Start time: "))
    end_time = int(input("Enter Length: "))
    export = input("Export output as GIF? [y/n]: ")
    if export.lower() == 'y':
        filename = input("Enter filename: ")
    end_time = start_time + end_time
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * milliseconds)
    pTime = 0
    detector = poseDetector()
    frames = []

    while True and cap.get(cv2.CAP_PROP_POS_MSEC) <= end_time * milliseconds:
    # while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            detector.findAngle(img, 28, 24, 27)
            # detector.findAngle(img, 24, 26, 28)
            # detector.findAngle(img, 26, 28, 32)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # show fps count
        cv2.putText(
            img, 'FPS:'+str(int(fps)), (120, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2,
        )
        cv2.imshow("Image", img)
        cv2.waitKey(1)
        frames.append(img)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

    if export.lower() == 'y':
        print("Saving GIF file")
        with imageio.get_writer("..//assets//model_runtime_output/{}.gif".format(filename), mode="I") as writer:
            for frame in frames:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                writer.append_data(rgb_frame)
        print("File saved")

if __name__ == "__main__":
    main()
