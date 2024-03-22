The early anticipation of non-ego-vehicle would caution the driver to take necessary action to avoid collision with non-ego-vehicle accidents. It would not only prevent an accident on the roads which result in costs to the car but also be deadly to the precious lives of the ego drivers. Recent research could anticipate accurately and early when accidents would occur for the task of accident anticipation. However, this research is only limited to ego-involved accidents. Therefore, the proposed model anticipates non-ego-accidents cases.

This deep neural network model that predicts the future trajectory of objects and anticipates non-ego-involved accidents only with object bounding box information. In the case of non-ego involved accidents, the model has been modified to handle multiple or at least two objects at a time. Therefore, the bounding box of two objects of accident frames is combined. 

The below figure shows the proposed traffic accident anticipation model for non-ego-involved cases. 
![image](https://github.com/heebahsaleem/scene_agnostic_non-ego/assets/16665306/e41d6e55-6b14-46a6-9b69-a00526592085)

This deep learning framework consists of four parts: (a) The first part has dashcam videos where videos are converted into frames. Then, each object is detected on those frames as bounding boxes and then tracked. b)
In the second stage, the prediction of the future trajectory of the object. The future trajectory is predicted using the object information from the past to the present. The input object information will be bounding box information only, and motion will be calculated during the model is running. Here, the RNN encoder-decoder structure is used. (c) The third part is the part in which the combination of two non-ego objects is created. After concatenating the two non-ego objects, these combinations are then matched with the annotation to check whether the two non-ego objects collide with each other. (d) In the last stage, the collided two non-ego objects are then input in this part. This part anticipates the accident between the two non-ego-vehicles using the future information predicted in the first part. This part anticipates the accident between the two non-egovehicles using the future information predicted in the first part. In this part, the probability of an accident between the two non-ego vehicles is calculated as an output. 

**Object Information:**
Object information consists of the location and size of the object, and the motion of the object. Pre-trained Mask R-CNN on the COCO dataset was used to extract the bounding boxes of objects in the frame. Deep-SORT algorithm is used for tracking each object and gives its own object ID by giving a threshold to the similarity between every object. Finally, the bounding box and object ID of each object are obtained in each frame. See below figure.

![image](https://github.com/heebahsaleem/scene_agnostic_non-ego/assets/16665306/35fcb084-aaeb-4e30-b89d-20953469eaa9)   ![image](https://github.com/heebahsaleem/scene_agnostic_non-ego/assets/16665306/e6411ba7-ddc3-41b3-95e8-0ce18848f5f5)

**Trajectory Prediction:**
The trajectory prediction part uses the RNN Encoder-Decoder structure. RNN decoder generates future bounding boxes with RNN. The encoded hidden states generated through the RNN encoders are used as the initial hidden states of the RNN decoder. 

**Hidden State Combination:**
The hidden state combination part will generate the concatenation of two non-ego objects. Hidden states of each object are extracted each of size 1024 from trajectory prediction part. The extracted hidden states from the trajectory prediction part are added with frame_id from the tracked objects pickle file. Frame_id is required to concatenate hidden states of a combination of two objects that should be aligned along frame_id. 

![image](https://github.com/heebahsaleem/scene_agnostic_non-ego/assets/16665306/b8ee5b9b-656b-4174-910a-68b56fcf2474)


The combination of hidden states of two objects is then matched with the annotation file to check whether the two objects collided with each other as shown in below figure. If two objects in a combination
collided with each other, then the label for the combination should be 1, otherwise 0. If the two combinations collided, then they are input to the accident anticipation part.

![image](https://github.com/heebahsaleem/scene_agnostic_non-ego/assets/16665306/b4d86125-3362-4cbe-80f0-056dfc3c0e93)

**Accident Anticipation:**
This part uses the two combinations of collided non-ego objects from (c) hidden state combination as input. "Accident probability (𝑝𝑡))" is calculated; accident probability means the possibility of a traffic accident between the two non-ego-vehicle. ATTC is obtained only for true positive (TP) data, which are predicted correctly as actual accidents

**Dataset:**

The dataset used in this study can be divided into 1) HEV-I dataset is used for training the neural network model of the Trajectory Prediction part and 2) DoTA dataset is used for training and testing the
neural network model of the Accident Anticipation part. 3) DoTA and CCD datasets used for testing model generality on the new dataset.

**My Full thesis:** https://library.gist.ac.kr/#/librarySearchDetails?book_no=490527
