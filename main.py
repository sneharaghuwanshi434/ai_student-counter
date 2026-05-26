import cv2

# Human detector initialize
hog = cv2.HOGDescriptor()

hog.setSVMDetector(
    cv2.HOGDescriptor_getDefaultPeopleDetector()
)

# Start camera
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize frame for better speed
    frame = cv2.resize(frame, (640, 480))

    # Detect humans
    boxes, weights = hog.detectMultiScale(frame)

    # Student counter
    count = 0

    # Draw rectangle
    for (x, y, w, h) in boxes:

        count += 1

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

    # Show count
    cv2.putText(
        frame,
        f"Students: {count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # Display output
    cv2.imshow("Student Counter", frame)

    # ESC key to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()