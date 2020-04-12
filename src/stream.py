import cv2, warnings, datetime

warnings.filterwarnings("ignore")
# os.environ["DYLD_PRINT_LIBRARIES"]='1'

cap = cv2.VideoCapture(0)
count = 0


def resize_image(size, im):
    desired_size = size
    old_size = im.shape[:2]  # old_size is in (height, width) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return new_im


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)
    frame = resize_image(400, frame)
    cv2.imwrite("images/A_" + str(count) + ".png", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    count += 1

# # When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
