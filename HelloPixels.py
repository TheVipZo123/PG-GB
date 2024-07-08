import cv2 as cv
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

def apply_filter(filter_name):
    if filter_name == "Grayscale Mean":
        show_image(img2)
    elif filter_name == "Grayscale Weighted Mean":
        show_image(img3)
    elif filter_name == "Red Channel":
        show_image(imgR)
    elif filter_name == "Green Channel":
        show_image(imgG)
    elif filter_name == "Blue Channel":
        show_image(imgB)
    elif filter_name == "Colorization":
        show_image(img4)
    elif filter_name == "Negative":
        show_image(img5)
    elif filter_name == "Sepia":
        show_image(img6)
    elif filter_name == "Invert Colors":
        show_image(img7)
    elif filter_name == "Gaussian Blur":
        show_image(img8)
    elif filter_name == "Edge Detection":
        show_image(img9)
    elif filter_name == "Sharpen":
        show_image(img10)
    elif filter_name == "Emboss":
        show_image(img11)
    elif filter_name == "Sobel X":
        show_image(img12)
    elif filter_name == "Sobel Y":
        show_image(img13)
    elif filter_name == "Bilateral Filter":
        show_image(img14)
    elif filter_name == "Median Filter":
        show_image(img15)
    elif filter_name == "Thresholding":
        show_image(img16)

def show_image(image):
    global displayed_image, imgtk
    displayed_image = image
    b, g, r = cv.split(image)
    img = cv.merge((r, g, b))
    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=im)
    panel.config(image=imgtk)
    panel.image = imgtk

def load_image():
    global img, img2, img3, imgR, imgG, imgB, img4, img5, img6, img7, img8, img9, img10, img11, img12, img13, img14, img15, img16, displayed_image
    filepath = filedialog.askopenfilename()
    if filepath:
        img = cv.imread(filepath)
        img2 = img.copy()
        img3 = img.copy()
        imgR = img.copy()
        imgG = img.copy()
        imgB = img.copy()
        img4 = img.copy()
        img5 = img.copy()
        img6 = img.copy()  # Initialize sepia filter image
        img7 = img.copy()  # Initialize invert colors filter image
        img8 = img.copy()  # Initialize Gaussian blur filter image
        img9 = img.copy()  # Initialize edge detection filter image
        img10 = img.copy()  # Initialize sharpen filter image
        img11 = img.copy()  # Initialize emboss filter image
        img14 = img.copy()  # Initialize bilateral filter image
        img15 = img.copy()  # Initialize median filter image
 

        for i in range(img.shape[0]):  # percorre as linhas
            for j in range(img.shape[1]):  # percorre as colunas
                # Grayscale Mean
                media = img.item(i, j, 0) * 0.33 + img.item(i, j, 1) * 0.33 + img.item(i, j, 2) * 0.33
                img2.itemset((i, j, 0), media)  # canal azul - B
                img2.itemset((i, j, 1), media)  # canal verde - G
                img2.itemset((i, j, 2), media)  # canal vermelho - R

                # Grayscale Weighted Mean
                mediaPond = img.item(i, j, 0) * 0.07 + img.item(i, j, 1) * 0.71 + img.item(i, j, 2) * 0.21
                img3.itemset((i, j, 0), mediaPond)  # canal azul - B
                img3.itemset((i, j, 1), mediaPond)  # canal verde - G
                img3.itemset((i, j, 2), mediaPond)  # canal vermelho - R

                # Red Channel
                imgR.itemset((i, j, 0), img.item(i, j, 2))  # canal azul - B
                imgR.itemset((i, j, 1), img.item(i, j, 2))  # canal verde - G
                imgR.itemset((i, j, 2), img.item(i, j, 2))  # canal vermelho - R

                # Green Channel
                imgG.itemset((i, j, 0), img.item(i, j, 1))  # canal azul - B
                imgG.itemset((i, j, 1), img.item(i, j, 1))  # canal verde - G
                imgG.itemset((i, j, 2), img.item(i, j, 1))  # canal vermelho - R

                # Blue Channel
                imgB.itemset((i, j, 0), img.item(i, j, 0))  # canal azul - B
                imgB.itemset((i, j, 1), img.item(i, j, 0))  # canal verde - G
                imgB.itemset((i, j, 2), img.item(i, j, 0))  # canal vermelho - R

                # Colorization
                img4.itemset((i, j, 0), img.item(i, j, 0) | cor[0])  # canal azul
                img4.itemset((i, j, 1), img.item(i, j, 1) | cor[1])  # canal verde
                img4.itemset((i, j, 2), img.item(i, j, 2) | cor[2])  # canal vermelho

                # Negative
                img5.itemset((i, j, 0), img.item(i, j, 0) ^ 255)
                img5.itemset((i, j, 1), img.item(i, j, 1) ^ 255)
                img5.itemset((i, j, 2), img.item(i, j, 2) ^ 255)

                # Sepia
                sepia_blue = min(255, 0.272 * img.item(i, j, 2) + 0.534 * img.item(i, j, 1) + 0.131 * img.item(i, j, 0))
                sepia_green = min(255, 0.349 * img.item(i, j, 2) + 0.686 * img.item(i, j, 1) + 0.168 * img.item(i, j, 0))
                sepia_red = min(255, 0.393 * img.item(i, j, 2) + 0.769 * img.item(i, j, 1) + 0.189 * img.item(i, j, 0))
                img6.itemset((i, j, 0), sepia_blue)
                img6.itemset((i, j, 1), sepia_green)
                img6.itemset((i, j, 2), sepia_red)

                # Invert Colors
                img7.itemset((i, j, 0), 255 - img.item(i, j, 0))
                img7.itemset((i, j, 1), 255 - img.item(i, j, 1))
                img7.itemset((i, j, 2), 255 - img.item(i, j, 2))

        # Gaussian Blur
        img8 = cv.GaussianBlur(img, (5, 5), 0)


        # Sharpen
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        img10 = cv.filter2D(img, -1, kernel)

        # Emboss
        kernel_emboss = np.array([[0, -1, -1],
                                  [1,  0, -1],
                                  [1,  1,  0]])
        img11 = cv.filter2D(img, -1, kernel_emboss)



        # Bilateral Filter
        img14 = cv.bilateralFilter(img, 9, 75, 75)

        # Median Filter
        img15 = cv.medianBlur(img, 5)



        show_image(img)

def save_image():
    if displayed_image is not None:
        filepath = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")])
        if filepath:
            cv.imwrite(filepath, displayed_image)

def load_sticker():
    global sticker
    filepath = filedialog.askopenfilename()
    if filepath:
        sticker = cv.imread(filepath, cv.IMREAD_UNCHANGED)  # Load sticker with alpha channel
        print("Sticker loaded")

def add_sticker_or_text(event):
    global displayed_image
    x, y = event.x, event.y
    x, y = int(x * displayed_image.shape[1] / panel.winfo_width()), int(y * displayed_image.shape[0] / panel.winfo_height())

    if event.num == 1:  
        if sticker is not None:
            sticker_h, sticker_w, _ = sticker.shape
            image_h, image_w, _ = displayed_image.shape

            if x + sticker_w > image_w or y + sticker_h > image_h:
                print("Sticker out of bounds")
                return

            # Add the sticker
            for i in range(sticker_h):
                for j in range(sticker_w):
                    if sticker[i, j, 3] != 0:  # Check the alpha channel for transparency
                        displayed_image[y + i, x + j] = sticker[i, j][:3]

            show_image(displayed_image)


# Initialize the Tkinter window
root = Tk()
root.title("Image Filter Application")

# Define the color for the colorization filter
cor = [255, 0, 255] # magenta

# Placeholder image to display initially
img = np.zeros((100, 100, 3), dtype=np.uint8)
img2 = img.copy()
img3 = img.copy()
imgR = img.copy()
imgG = img.copy()
imgB = img.copy()
img4 = img.copy()
img5 = img.copy()
img6 = img.copy()  # Initialize sepia filter image
img7 = img.copy()  # Initialize invert colors filter image
img8 = img.copy()  # Initialize Gaussian blur filter image
img9 = img.copy()  # Initialize edge detection filter image
img10 = img.copy()  # Initialize sharpen filter image
img11 = img.copy()  # Initialize emboss filter image

img14 = img.copy()  # Initialize bilateral filter image
img15 = img.copy()  # Initialize median filter image


displayed_image = img
sticker = None  # Initialize sticker as None

# Display the initial image
b, g, r = cv.split(img)
img_display = cv.merge((r, g, b))
im = Image.fromarray(img_display)
imgtk = ImageTk.PhotoImage(image=im)
panel = Label(root, image=imgtk)
panel.image = imgtk
panel.pack(side="top", fill="both", expand="yes")


panel.bind("<Button-1>", add_sticker_or_text)
panel.bind("<Button-3>", add_sticker_or_text)

# Create buttons for each filter
buttons_frame = Frame(root)
buttons_frame.pack(side="bottom", fill="x")

Button(buttons_frame, text="Load Image", command=load_image).pack(side="left")
Button(buttons_frame, text="Save Image", command=save_image).pack(side="left")
Button(buttons_frame, text="Load Sticker", command=load_sticker).pack(side="left")
Button(buttons_frame, text="Grayscale Mean", command=lambda: apply_filter("Grayscale Mean")).pack(side="left")
Button(buttons_frame, text="Grayscale Weighted Mean", command=lambda: apply_filter("Grayscale Weighted Mean")).pack(side="left")
Button(buttons_frame, text="Red Channel", command=lambda: apply_filter("Red Channel")).pack(side="left")
Button(buttons_frame, text="Green Channel", command=lambda: apply_filter("Green Channel")).pack(side="left")
Button(buttons_frame, text="Blue Channel", command=lambda: apply_filter("Blue Channel")).pack(side="left")
Button(buttons_frame, text="Colorization", command=lambda: apply_filter("Colorization")).pack(side="left")
Button(buttons_frame, text="Negative", command=lambda: apply_filter("Negative")).pack(side="left")
Button(buttons_frame, text="Sepia", command=lambda: apply_filter("Sepia")).pack(side="left")
Button(buttons_frame, text="Invert Colors", command=lambda: apply_filter("Invert Colors")).pack(side="left")
Button(buttons_frame, text="Gaussian Blur", command=lambda: apply_filter("Gaussian Blur")).pack(side="left")

Button(buttons_frame, text="Sharpen", command=lambda: apply_filter("Sharpen")).pack(side="left")
Button(buttons_frame, text="Emboss", command=lambda: apply_filter("Emboss")).pack(side="left")

Button(buttons_frame, text="Bilateral Filter", command=lambda: apply_filter("Bilateral Filter")).pack(side="left")
Button(buttons_frame, text="Median Filter", command=lambda: apply_filter("Median Filter")).pack(side="left")

root.mainloop()
