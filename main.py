import cv2 as cv
import numpy as np

cap = cv.VideoCapture('test_video.mp4')


def make_points(image, average):
    slope, y_int = average

    y1 = image.shape[0]
    y2 = int(y1 * (6 / 8))
    x1 = int((y1 - y_int) / slope)
    x2 = int((y2 - y_int) // slope)

    return np.array([x1, y1, x2, y2])


if __name__ == "__main__":

    left_avg = None
    right_avg = None

    while cap.isOpened():
        # READ VIDEO FILE
        ret, frame = cap.read()

        if ret:
            # GRAYSCALE IMAGE
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            gray = cv.GaussianBlur(gray, (3, 3), 0)

            # CONVERT IMAGE TO HSV TO DETECT YELLOW
            img_hsv = cv.cvtColor(frame, cv.COLOR_RGB2HSV)
            img_hsv = cv.GaussianBlur(img_hsv, (5, 5), 0)

            # SET THRESHOLDS FOR YELLOW DETECTION
            lower_yellow = np.array([20, 100, 100], dtype="uint8")
            upper_yellow = np.array([30, 255, 255], dtype="uint8")

            # APPLY MASK TO ISOLATE YELLOW AND WHITE
            mask_yellow = cv.inRange(img_hsv, lower_yellow, upper_yellow)
            mask_white = cv.inRange(gray, 200, 255)
            mask_yw = cv.bitwise_or(mask_white, mask_yellow)
            mask_yw_image = cv.bitwise_and(gray, mask_yw)

            # DETECT EDGES FROM FILTERED IMAGE
            edges = cv.Canny(mask_yw_image, 50, 150)

            # ISOLATE VIEW OF LANE
            height, width, channels = frame.shape
            triangle = np.array([
                [(200, height - 20), ((width / 2) + 10, 520), (width - 350, height - 20)]
            ])

            mask = np.zeros_like(edges)
            mask = cv.fillPoly(mask, np.int32([triangle]), 255)
            mask = cv.bitwise_and(edges, mask)

            # DETECT LINES FROM FOCUSED VIEW OF LANES
            lines = cv.HoughLinesP(mask, 1, np.pi/180, 25, minLineLength=10, maxLineGap=600)

            if lines is not None:
                left = []
                right = []

                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    parameters = np.polyfit((x1, x2), (y1, y2), 1)
                    slope = parameters[0]
                    y_int = parameters[1]

                    if slope < 0:
                        left.append((slope, y_int))
                    else:
                        right.append((slope, y_int))

                if len(right) > 0:
                    right_avg = np.average(right, axis=0)
                if len(left) > 0:
                    left_avg = np.average(left, axis=0)

                left_line = make_points(frame, left_avg)
                right_line = make_points(frame, right_avg)
                filtered_lines = np.array([left_line, right_line])

                lines_image = np.zeros_like(frame)
                for line in filtered_lines:
                    x1, y1, x2, y2 = line
                    cv.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 10)

            cv.imshow('Frame', frame)
            cv.imshow('Mask', mask)
        else:
            cap = cv.VideoCapture('test_video.mp4')

        if cv.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
