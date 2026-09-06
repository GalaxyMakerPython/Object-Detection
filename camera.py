import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Colors


model = YOLO(r"C:\Users\RSKALA\Desktop\Python Files\best.pt")

custom_palette = Colors()
custom_palette.palette = [
    (255, 0, 0),    
    (0, 0, 255),    
    (0, 255, 0),    
]

cap = cv2.VideoCapture(0)

while True:
    itr, frame = cap.read()  
    if not itr:
        break

    
    result = model.predict(source=frame, imgsz=320, verbose=False) 
    
    
    frame_final = result[0].plot()

    cv2.imshow("a", frame_final)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()