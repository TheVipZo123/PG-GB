import numpy as np
import cv2 as cv
from tkinter import *
from tkinter import ttk, filedialog

def apply_filter(filter_name):
    global current_filter
    current_filter = filter_name

def process_frame(frame):
    global current_filter
    if current_filter == "Canny":
        return cv.Canny(frame, 50, 100)
    elif current_filter == "Grayscale":
        return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    elif current_filter == "Sepia":
        frame = np.array(frame, dtype=np.float64)
        frame = cv.transform(frame, np.matrix([[0.272, 0.534, 0.131],
                                               [0.349, 0.686, 0.168],
                                               [0.393, 0.769, 0.189]]))
        frame[np.where(frame > 255)] = 255
        return np.array(frame, dtype=np.uint8)
    elif current_filter == "Invert":
        return cv.bitwise_not(frame)
    elif current_filter == "Blur":
        return cv.GaussianBlur(frame, (15, 15), 0)
    elif current_filter == "Edge Detection":
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        return cv.Canny(gray, 100, 200)
    elif current_filter == "Sharpen":
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        return cv.filter2D(frame, -1, kernel)
    elif current_filter == "Emboss":
        kernel = np.array([[0, -1, -1],
                           [1,  0, -1],
                           [1,  1,  0]])
        return cv.filter2D(frame, -1, kernel)
    elif current_filter == "Threshold":
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
        return cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
    elif current_filter == "None":
        return frame

    return frame

def save_frame():
    global current_frame
    if current_frame is not None:
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")])
        if file_path:
            cv.imwrite(file_path, current_frame)

def update_frame():
    global current_frame
    ret, frame = capture.read()
    if frame is None:
        return
    current_frame = process_frame(frame)
    cv.imshow('Video', current_frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        root.destroy()
        capture.release()
        cv.destroyAllWindows()
    else:
        root.after(10, update_frame)

# Initialize video capture
capture = cv.VideoCapture(0)
if not capture.isOpened():
    print('Unable to open camera')
    exit(0)

# Initialize Tkinter window
root = Tk()
root.title("Real-Time Video Filters")

# Create buttons for filters
filters = ["None", "Canny", "Grayscale", "Sepia", "Invert", "Blur", "Edge Detection", "Sharpen", "Emboss", "Threshold"]
current_filter = "None"

for filter_name in filters:
    button = Button(root, text=filter_name, command=lambda fn=filter_name: apply_filter(fn))
    button.pack(side=LEFT)

# Create a button to save the frame
save_button = Button(root, text="Save Frame", command=save_frame)
save_button.pack(side=LEFT)

# Start updating the frames
current_frame = None
update_frame()

# Start Tkinter main loop
root.mainloop()
