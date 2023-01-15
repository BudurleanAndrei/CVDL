import cv2
import random
import numpy as np

from Constants import NUMBER_OF_QUESTIONS, NUMBER_OF_ANSWERS, NUMBER_OF_ERROR_ANSWERS, word_dict


def generateRandomAnswers(image):
    full_blob = cv2.imread("Full_Blob.jpg")
    empty_blob = cv2.imread("Empty_Blob.jpg")


    blobs = image[248: 668, 120: 260]
    height = int(blobs.shape[0] / NUMBER_OF_QUESTIONS)
    width = int(blobs.shape[1] / NUMBER_OF_ANSWERS)
    # print(blobs.shape)
    # print(height)
    # print(width)

    # cv2.imshow("Image", image)
    # cv2.waitKey()

    for i in range(1, NUMBER_OF_QUESTIONS + 1):
        index = random.randint(1, NUMBER_OF_ANSWERS)
        row = blobs[(i - 1) * height: i * height, 0: blobs.shape[1]]
        # cv2.imshow("Image", row)
        # cv2.waitKey()

        for j in range(1, NUMBER_OF_ANSWERS + 1):
            answer = row[0: row.shape[0], (j - 1) * width: j * width]
            # cv2.imshow("Image", answer)
            # cv2.waitKey()
            if j == index:
                blobs[(i - 1) * height: i * height, (j - 1) * width: j * width] = full_blob
            else:
                blobs[(i - 1) * height: i * height, (j - 1) * width: j * width] = empty_blob

    image[248: 668, 120: 260] = blobs

    # cv2.imshow("Image", image)
    # cv2.waitKey()


def generateRandomErrors(image):
    full_blob = cv2.imread("Full_Blob.jpg")
    empty_blob = cv2.imread("Empty_Blob.jpg")

    blobs = image[248: 668, 266: 289]
    # cv2.imshow("Image", blobs)
    # cv2.waitKey()

    height = int(blobs.shape[0] / NUMBER_OF_QUESTIONS)
    width = blobs.shape[1]
    # print(blobs.shape)
    # print(height)
    # print(width)

    errors = list()
    for i in range(1, NUMBER_OF_QUESTIONS + 1):
        index = random.randint(0, 9)

        if index == 0:
            errors.append(i)
            blobs[(i - 1) * height: i * height, 0: width] = full_blob
        else:
            blobs[(i - 1) * height: i * height, 0: width] = empty_blob

    image[248: 668, 266: 289] = blobs
    # cv2.imshow("Image", image)
    # cv2.waitKey()
    return errors

def generateRandomErrorAnswers(image, errors):
    letters = list()
    blank = np.zeros([28, 28, 3], dtype=np.uint8)
    blank.fill(255)
    blank = cv2.copyMakeBorder(blank, top=1, bottom=1, left=1, right=1, borderType=cv2.BORDER_CONSTANT)

    for value in word_dict.values():
        img = cv2.imread("Datasets/Letters/" + value + ".jpeg")
        img = cv2.copyMakeBorder(img, top=1, bottom=1, left=1, right=1, borderType=cv2.BORDER_CONSTANT)
        letters.append(img)
    letters = letters[:NUMBER_OF_ANSWERS]

    squares = image[256: 668, 328: 394]
    squares = cv2.resize(squares, (66, 420))

    height = int(squares.shape[0] / NUMBER_OF_QUESTIONS)
    width = int(squares.shape[1] / NUMBER_OF_ERROR_ANSWERS)

    for i in range(1, NUMBER_OF_QUESTIONS + 1):
        row = squares[(i - 1) * height: i * height, 0: squares.shape[1]]

            # cv2.imshow("Squares", row)
            # cv2.waitKey()
        for j in range(1, NUMBER_OF_ERROR_ANSWERS + 1):
            index = random.randint(0, len(letters) - 1)

            if i in errors:
                row[0: row.shape[0], (j - 1) * width: j * width] = cv2.resize(letters[index], (22, 21))
            else:
                row[0: row.shape[0], (j - 1) * width: j * width] = cv2.resize(blank, (22, 21))
        squares[(i - 1) * height: i * height, 0: squares.shape[1]] = row

    squares = cv2.resize(squares, (66, 412))
    image[256: 668, 328: 394] = squares


def generateImage(name):
    image = cv2.imread("TestExample.png")
    image = cv2.resize(image, (600, 800))

    # cv2.imshow("Image", image)
    # cv2.waitKey()

    generateRandomAnswers(image)
    errors = generateRandomErrors(image)
    generateRandomErrorAnswers(image, errors)

    cv2.imwrite("Datasets/Tests/" + name + ".png", image)

    # cv2.imshow("Image", image)
    # cv2.waitKey()


def generateNImages(n=100):
    for i in range(1, n + 1):
        generateImage(str(i))


generateNImages()
