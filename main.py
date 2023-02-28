import cv2
import math
import argparse
def facedetection(net, frame, conf_threshold=0.7):
    f=frame.copy()
    height=f.shape[0]
    width=f.shape[1]
    blob=cv2.dnn.blobFromImage(f, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections=net.forward()
    boxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*width)
            y1=int(detections[0,0,i,4]*height)
            x2=int(detections[0,0,i,5]*width)
            y2=int(detections[0,0,i,6]*height)
            boxes.append([x1,y1,x2,y2])
            cv2.rectangle(f, (x1,y1), (x2,y2), (0,255,0), int(round(height/150)), 8)
    return f,boxes
p=argparse.ArgumentParser()
p.add_argument('--image')
args=p.parse_args()
faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
list_of_age=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
list_of_gender=['Male','Female']
faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)
video=cv2.VideoCapture(args.image if args.image else 0)
padding=20
while cv2.waitKey(1)<0:
    hasFrame,frame=video.read()
    if not hasFrame:
        cv2.waitKey()
        break
    result,boxes=facedetection(faceNet,frame)
    if not boxes:
        print("No face detected")
    for box in boxes:
        face=frame[max(0,box[1]-padding):
                   min(box[3]+padding,frame.shape[0]-1),max(0,box[0]-padding)
                   :min(box[2]+padding, frame.shape[1]-1)]
        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=list_of_gender[genderPreds[0].argmax()]
        print(f'Gender: {gender}')
        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=list_of_age[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')
        cv2.putText(result, f'{gender}, {age}', (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting age and gender", result)
