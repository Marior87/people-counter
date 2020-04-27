# Project Write-Up

This project was made as one of the requirements in order to obtain Intel(R) Edge AI for IoT Developers Nanodegree program at Udacity. In this Write Up, I will explain some concepts related with OpenVINO usage and some considerations from the user side.

## Explaining Custom Layers

When loading a model inside an OpenVINO application, chances are that the model has some complexity, in the form of layers (operations). This could create situations where is not possible for the Model Optimizer to create an appropriate direct Intermediate Representation.

In that case, OpenVINO offers different options depending on model original framework (TensorFlow, Caffe, MXNet, etc), available devices and user preferences, to handle those not recognized operations as Custom Layers.

At the moment the Model Optimizer tries to convert a model, it will give more relevance to preserve any defined Custom Layer, which means that if a layer is defined as custom (with some internal logic), it will be used even if such layer is already supported. 
The simplified workflow is as follows:
  1. When a model is loaded, will check first if it is defined as a Custom Layer, depending of the framework, this could be either the layer is already described in a Custom Layers file, or as an extension for the Model Optimizer. If not, then it will check if the layer could be described using default configurations already available with OpenVINO.

  2. There are cases when layer’s complexity (or user preferences) makes it not possible to create a representation that is suitable for the Inference Engine. In those scenarios, we can redirect the computation effort to the model original framework. This has some disadvantages, like that the device where the model will run must have the framework installed and, for sure, it will not be as efficient as if the model was completely represented as a full OpenVINO Intermediate Representation.

The result of the Model Optimizer conversion of a model with (or without) Custom Layers should be Intermediate Representation files (.xml and .bin). But for the application, in order to be able to run, the Inference Engine needs also to be fed with relevant extensions regarding model layers, including custom ones. These extensions are device-related, and it is also possible to have an application running on multiple different devices, each one handling different extensions.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was...

The size of the model pre- and post-conversion was...

The inference time of the model pre- and post-conversion was...

## Assess Model Use Cases

People counter app has many potential uses, some of them are:

  * To guarantee a place is not overcrowded, or there are some limitations related with the quantity of people allowed to be there. Like in these days with COVID19 issues, places like grocery stores and bank would need to avoid people concentrations in closed spaces.
  
  * For Urban Planning, like checking the number of people that uses determined streets, will help Urban Planners create projects in order to improve city life quality. It is also possible to modify the people counter app to count vehicles (with minor changes), giving them more valuable information of the city dynamics.
  * Retail stores could create people traffic heat-maps to check where people prefer to be in the store, providing useful information for targeted marketing, products positioning and even how many people visited the store.
  * In emergency scenarios, it will be useful to know how many people is inside a place (building, factory, etc) as an indicator for possible rescue operations.
  * For security reasons, it could be an indicator of some problems happening if someone spends too much time in a given place, from factories (hazardous zones) to banks (time spent in ATMs).

These or any other applications should also consider that the app is not perfect, which means some conclusions should not be taken expecting this. For example, a person could enter with a blanket covering him and the app (as it is presented here) will not make a detection. It is possible to include other elements to avoid this situations, but as for today, there are not perfect people detection models.


## Assess Effects on End User Needs

As with any computer vision application, and this is not by any means an issue only with OpenVINO, they are very affected by conditions present during image capturing.

When we develop a computer vision application, we must consider as many different scenarios as possible, small things like lighting (for example, if the camera is placed outdoors, we must evaluate poor lighting conditions during nights) can dramatically diminish application performance. Handling bad quality images should also be taken into account, but for most cases also high quality images are not available or neither are convenient. This point is very model dependent, as most of them are trained under very specific situations. The main idea is to feed them with the minimum conditions necessary for acceptable accuracy.

Also, as stated in the previous section, model accuracy should be taken with caution, it is necessary to create some inside application logic in order to reduce as much as possible any model error, and some times it is not only related with model complexity. For example, for my application I tested an SSD Inception model and SSDLite one, the former was almost 5 times bigger than the later, but they failed in the same situations (a person standing very still), that is why, for other performance considerations, I chose SSDLite and included some application logic to cope with those misdetections.

Other important considerations imply the nature of IoT devices, which need to work in very constrained environments, as opposed to applications running on a laptop or in the cloud. One of the most relevant ones is network usage.

Any network use means that the device should use some of it resources for sending information, resources like processing capabilities (even if they are small) and power consumption, maybe the most constrained resources any IoT device has. Regarding this, the application logic should be designed to only send relevant information through the network, especially for computer vision applications where it could be necessary to send images.

On the other hand, IoT devices can leverage valuable information where cloud services could be too expensive to use, not preferred due to security reasons or not even available. For example, regarding pricing, as for today, Google Cloud Vision API has a minimum price for object detection of 1.5$ per 1000 images, to put in context, this project video example has around 1300 frames in total and it is only 2 minutes and 20 seconds long, which means that in the conditions I developed it (detection on each frame), running 10 times the video, for a relative 23.5 minutes video duration, will cost around 19.5$, and for sure inference time will not be as fast as with the IoT device.

However, this doesn’t mean that cloud computing is always worse, we need to consider the problem we need to solve and, having all this options available, take the one (or maybe both) that suits better our conditions. 
