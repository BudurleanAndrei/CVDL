from PIL import Image
import numpy as np
import cv2
from keras.optimizers import Adam
from keras.preprocessing import image

from Constants import *
from keras.models import load_model

def processImage(imagePath):
    image = Image.open(imagePath)
    image = image.resize((HEIGHT, WIDTH))
    image = image.crop((PHOTO_LEFT, PHOTO_TOP, PHOTO_RIGHT, PHOTO_BOTTOM))
    imgW, imgH = image.size
    answers = image.crop((ADJUST_ANSWERS_LEFT, ADJUST_ANSWERS_TOP, imgW - ADJUST_ANSWERS_RIGHT, imgH - ADJUST_ANSWERS_BOTTOM))
    error = image.crop((imgW - ADJUST_ERROR_LEFT, ADJUST_ERROR_TOP, imgW - ADJUST_ERROR_RIGHT, imgH - ADJUST_ERROR_BOTTOM))
    errorAnswers = image.crop((imgW - ADJUST_ERROR_ANSWERS_LEFT, ADJUST_ERROR_ANSWERS_TOP, imgW - ADJUST_ERROR_ANSWERS_RIGHT, imgH - ADJUST_ERROR_ANSWERS_BOTTOM))
    # print(answers.size)
    # print(error.size)
    # print(errorAnswers.size)

    # image.show()
    # answers.show()
    # error.show()
    # errorAnswers.show()
    return answers, error, errorAnswers


def checkAnswers(answersImage):
    rowWidth = answersImage.size[0] / NUMBER_OF_ANSWERS
    rowHeight = answersImage.size[1] / NUMBER_OF_QUESTIONS
    processedAnswers = dict()

    detector = cv2.SimpleBlobDetector_create()
    keypoint = detector.detect(np.asarray(answersImage))
    im_with_keypoints = cv2.drawKeypoints(np.asarray(answersImage), keypoint, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    for i in range(1, NUMBER_OF_QUESTIONS + 1):
        processedAnswers[i] = '-'
        row = im_with_keypoints[int((i - 1) * rowHeight) : int(i * rowHeight), 0 : int(answersImage.size[0])]

        for j in range(1, NUMBER_OF_ANSWERS + 1):
            answer = row[0 : int(rowHeight), int((j - 1) * rowWidth) : int(j * rowWidth)]

            # In case the photos have 4 layers on the last dimension (that would be alpha)
            # lower_red = np.array([10, 10, 20, 255])
            # upper_red = np.array([60, 60, 255, 255])
            lower_red = np.array([10, 10, 20])
            upper_red = np.array([60, 60, 255])
            mask = cv2.inRange(answer, lower_red, upper_red)

            nr_red_pix = np.sum(mask == 255)

            if nr_red_pix > 20:
                processedAnswers[i] = chr(j + 64)

    # print(processedAnswers)
    return processedAnswers


def checkErrors(errorsImage):
    rowHeight = errorsImage.size[1] / NUMBER_OF_QUESTIONS
    processedErrors = dict()
    # errorsImage.show()

    detector = cv2.SimpleBlobDetector_create()
    keypoint = detector.detect(np.asarray(errorsImage))
    im_with_keypoints = cv2.drawKeypoints(np.asarray(errorsImage), keypoint, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow("Keypoints", im_with_keypoints)
    # cv2.waitKey()

    for i in range(1, NUMBER_OF_QUESTIONS + 1):
        processedErrors[i] = 'F'
        row = im_with_keypoints[int((i - 1) * rowHeight): int(i * rowHeight) - 5, 0: int(errorsImage.size[0])]

        # In case the photos have 4 layers on the last dimension (that would be alpha)
        # lower_red = np.array([10, 10, 20, 255])
        # upper_red = np.array([60, 60, 255, 255])

        lower_red = np.array([0, 0, 10])
        upper_red = np.array([10, 10, 255])
        mask = cv2.inRange(row, lower_red, upper_red)

        if 255 in mask:
            processedErrors[i] = "T"

    # print(processedErrors)
    return processedErrors


def checkErrorAnswers(errors, errorAnswersImage):
    errorAnswersImage = errorAnswersImage.resize((errorAnswersImage.size[0], errorAnswersImage.size[1] + 10))
    rowWidth = errorAnswersImage.size[0] / NUMBER_OF_ERROR_ANSWERS
    rowHeight = errorAnswersImage.size[1] / NUMBER_OF_QUESTIONS
    processedAnswers = dict()

    model = load_model(r'model_hand.h5')
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    for i in range(1, NUMBER_OF_QUESTIONS + 1):
        processedAnswers[i] = list()
        row = errorAnswersImage.crop((0, (i - 1) * rowHeight, errorAnswersImage.size[0], i * rowHeight))

        for j in range(1, NUMBER_OF_ERROR_ANSWERS + 1):
            answer = row.crop(((j - 1) * rowWidth, 0, j * rowWidth, row.size[1]))

            if errors[i] == 'T':
                img = cv2.resize(np.asarray(answer), (140, 140))
                img_copy = img.copy()

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (400,440))
                img_copy = cv2.GaussianBlur(img_copy, (7,7), 0)
                img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
                _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

                img_final = cv2.resize(img_thresh, (28,28))
                img_final = np.reshape(img_final, (1,28,28,1))
                img_pred = word_dict[np.argmax(model.predict(img_final, verbose=0))]

                # cv2.imshow("Answer", img)
                # cv2.waitKey()
                # answer = answer.reshape((1, 28, 28, 1))
                if img_pred == 'G':
                    img_pred = 'C'
                if img_pred == 'P':
                    img_pred = 'B'
                processedAnswers[i].append(img_pred)

    return processedAnswers


def evalAllTests():
    results = list()
    allowed_answers = list(word_dict.values())[:NUMBER_OF_ANSWERS]
    for i in range(1, 2):
        correct_answers = 0
        answ = list()
        answersImage, errorImage, errorAnswersImage = processImage("Datasets/Tests/" + str(i) + ".png")
        processedAnswers = checkAnswers(answersImage)
        processedErrors = checkErrors(errorImage)
        processedErrorAnswers = checkErrorAnswers(processedErrors, errorAnswersImage)

        for j in range(1, NUMBER_OF_QUESTIONS + 1):
            if processedErrors[j] == 'T':
                answer = None
                for val in processedErrorAnswers[j]:
                    if val in allowed_answers:
                        answer = val
                answ.append(answer)
                if answer is not None and answer == ANSWERS[j - 1]:
                    correct_answers += 1
            else:
                answ.append(processedAnswers[j])
                if processedAnswers[j] == ANSWERS[j - 1]:
                    correct_answers += 1
        grade = correct_answers / NUMBER_OF_QUESTIONS * 10
        print(answ)
        print(correct_answers)
        results.append(grade)
    print(results)


evalAllTests()
