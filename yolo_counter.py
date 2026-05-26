from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Start camera
cap = cv2.VideoCapture(0)

while True:

    # Read frame
    ret, frame = cap.read()

    if not ret:
        break

    # Run detection
    results = model(frame, conf=0.3, verbose=False)

    count = 0

    for result in results:

        boxes = result.boxes

        for box in boxes:

            cls = int(box.cls[0])

            # Person class
            if cls == 0:

                count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw rectangle
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0,255,0),
                    2
                )

                # Label
                cv2.putText(
                    frame,
                    "Student",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2
                )

    # Show count
    cv2.putText(
        frame,
        f"Students: {count}",
        (20,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        2
    )

    # Display window
    cv2.imshow("YOLO Student Counter", frame)

    # ESC key
    if cv2.waitKey(1) == 27:
        break

# Release camera
cap.release()

# Close windows
cv2.destroyAllWindows()