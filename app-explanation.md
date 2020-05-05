# Explaining the logic used in this project:

## Case:

We were provided with a video file showing people entering a place from the lower side of the footage, checking a list and then going out to the right (see resources/Pedestrian_Detect_2_1_1.mp4). Our assignment was, in short, to make a people counter that could detect when someone enters the frame, when leaves and keep track of some statistics.

## Solution approach:

I downloaded and converted some TensorFlow models into IR (check README.md for more details about how to do it):

* SSDLite MobileNet v2
* SSD Inception v2
* Faster RCNN Inception v2

IR models are stored under tfmodels folder.

### SSD Models:
During testing, both SSD models have similiar accuracy, although SSD Inception model was significantly slower. They tend to not classify properly when a person was standing too still and not facing the camera.

For them, I decided to create a logic that could do the job, using different considerations and taking advantage of the similar flow all people showed:
* At time zero, the frame is divided in two halves, with an imaginary line set at LOWER_HALF, in our case at 0.7*height (this value was reached by trial and error).
* For any detection, calculate the corresponding bounding box centroid.
* As people enters from the lower side, when there is a detection the centroid will be located in the lower half, changing the status_lower_half to 'True'.
* Suqsequent detections will move up the centroid, until reaching the LOWER_HALF boundary and set now status_lower_half to 'False' and status_upper_half to 'True'.
* At this point, RIGHT_HALF (0.8*width in our case) will be the boundary that will affect the logic. When a centroid crosses this imaginary line, it will be considered that the person leaves the frame, which will trigger the calculation of relevant statistics and turning status_upper_half to 'False', which is our initial point.

### Faster RCNN:
With this model, the logic was easier because there were fewer misdetections. It just has flags for detection moments and uses a frames_for_quit value to counter any consecutive misdetections. This value is also considered when calculating total time.

On the other hand, this model was **much slower** than any of the SSD models, making it not appropriate at all, at least considering my hardware.

Regarding this, this application works with both type of models (even tho they have different workflows and model interpretation), just be sure that for Faster RCNN models the word 'faster' is somewhere in the args.input file name.

### Conclusion:
Adapting to real life situations could mean that we have to be flexible in order to obtain desired results. Each scenario is different and should be treated accordingly. Exploring diffrente model options is an important task for any AI application, but we need to be prepared for not finding "the perfect model".