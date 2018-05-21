import cv2


def main():
    filename = 'Ex'
    img = cv2.imread('{}.png'.format(filename))
    #img = img[60:420, 84:570]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)),
        iterations=1)
    mask = cv2.erode(
        mask,
        cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)),
        iterations=1)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)))
    img_anot = cv2.bitwise_and(img, img, mask=mask)
    img_final = cv2.inpaint(img, cv2.bitwise_not(mask), 3, cv2.INPAINT_TELEA)
    cv2.imwrite('{}_mask.png'.format(filename), mask)
    cv2.imwrite('{}_anot.png'.format(filename), img_anot)
    cv2.imwrite('{}_final.png'.format(filename), img_final)


if __name__ == '__main__':
    main()
