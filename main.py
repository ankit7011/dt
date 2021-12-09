import numpy as np
import cv2
import os



# KNN 

def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

def KNN(train,qp,k=5):
    vals=[]
    m=train.shape[0]
    x=train[ : , : -1]
    y=train[:,-1]
    for i in range(m):
        d=dist(qp,x[i])
        vals.append((d,y[i]))
    
    vals=sorted(vals)
    vals=vals[:k]
    vals=np.array(vals)

    new_vals=np.unique(vals[:,-1],return_counts=True)

    index=new_vals[1].argmax()
    pred=new_vals[0][index]

    return pred


cap=cv2.VideoCapture(0)


Face_cascade=cv2.CascadeClassifier("har.xml")

skip=0

dataset_path='./data_set/'


face_data=[]
label=[]

class_id=0
names={}


for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        print("loaded " + fx)
        names[class_id]=fx[:-4]
        data_item=np.load(dataset_path+fx)
        face_data.append(data_item)

        target=class_id * np.ones((data_item.shape[0]))
        class_id+=1
        label.append(target)

face_dataset=np.concatenate(face_data,axis=0)
face_labels=np.concatenate(label,axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)

trainset=np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)


while True: 
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    if(ret==False):
        continue
    faces=Face_cascade.detectMultiScale(frame,1.3,5)



    for x,y,w,h in faces:
        offset=0
        # face_section=frame[y-offset : y+h+offset , x-offset , x+w + offset]
        face_section=frame
        face_section=cv2.resize(face_section,(100,100))

        pred=KNN(trainset,face_section.flatten())

        pred_name=names[int(pred)]
        cv2.putText(frame,pred_name,(x,y-offset),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
    
    cv2.imshow("Faces ", frame)
    

    key_pressed=cv2.waitKey(1)&0xFF
    if(key_pressed==ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
