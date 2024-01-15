# Determining CVS from Lateral Cephalograms
## ABSTRACT

Traditionally, lateral cephalograms have been employed in the diagnostic and treatment planning of orthodontic patients. The degree of ossification in bone is referred to as skeletal maturation. Since the wide spectrum in the timing and duration of the pubertal growth spurt and other developmental phases, chronological age is inaccurate for assessing developmental state. Doctors and clinicians must examine lateral cephalograms to determine the orthodontic procedure to be provided to the patient. Doctors manually classify the cephalograms into six stages based on the shape of the bones. Each of the classes has distinct diagnostic procedures. This procedure is time intensive and has the disadvantage of being subject to random and systematic error, most errors occur in landmark identification, which is based on observer experience, landmark definition, and image density and sharpness and hence requires expertise. Therefore, automating this process would provide a more reliable platform. We plan on training a deep learning model to successfully classify the lateral cephalograms into one of the six CVS stages: Initiation (CVS 1), Acceleration (CVS 2), Transition, (CVS 3), Deceleration (CVS 4), Maturation (CVS 5), Completion (CVS 6).

## INTRODUCTION
In the field of dental and maxillofacial radiology (DMFR), reports on AI and deep learning
models used for diagnostic purposes and treatment planning cover a wide range of clinical
applications, including automated localization of craniofacial anatomical structures/pathological changes, classification of maxillofacial cysts and/or tumours, and
diagnosis of caries and periodontal lesions. Lateral cephalometry has been widely used for
skeletal classification in orthodontic diagnosis and treatment planning. However, this
conventional system, requiring manual tracing of individual landmarks, contains possible
errors of inter and intra variability and is highly time-consuming.
Our project's goal is to create and evaluate an algorithm for determining cervical vertebral
stages in lateral cephalograms, which will represent an individual's growth and
development. Each cephalogram must be categorized into one of six phases. Each stage
requires a unique set of diagnostic procedures to be performed on patients. We are currently
acquiring lateral cephalograms collected by a Post Graduate student at the Dayananda
Sagar Dental College. At present, the number of images received are ~670. Around 570
images have been undergone different pre-processing techniques.

## METHODOLOGY

### 1. DATASET ACQUISITION
The data is being collected by a Post Graduate Student in Dayananda Sagar Dental
College. It consists of 800 images of Lateral Cephalograms.
- A considerable amount of time was dedicated to the collection of cephalograms.
- The raw data in SVG format was converted to JPEG format during the process
of digitalization.
- The classification of the data was done manually, and the respective labels were
entered into a CSV file.
- The lateral cephalograms are categorized according to their CVS stages. There
are 6 CVS stages: Initiation (CVS 1), Acceleration (CVS 2), Transition, (CVS 3),
Deceleration (CVS 4), Maturation (CVS 5), Completion (CVS 6).

### 2. DATA PRE-PROCESSING
The below steps were finalized after trying a series of various preprocessing techniques
such as thresholding, contours, erosion, dilation, etc.
- Resizing: The images are resized according to the requirement. The original
images were of 1804px width and variable height. The images were resized to
1800x2100 based on the mean height of all the images.
- Cropping: The images were cropped based on the region of interest (ROI) i.e., C2,
C3, C4 cervical vertebrae. The resulting images were of size 500x500.
- Frequency Check: The frequency of each CVS stage must be the same to avoid
overfitting. So, we cap the frequency of each stage to 100 images while training the
model.
- CLAHE: It is an image enhancement technique that is used for contrast
enhancement. It helps make the image clearer, which in turn increases the accuracy
of the prediction. We have applied CLAHE with a clipLimit of 2 for all the images.

### 3. MODELS
Currently testing with various pre-trained models such as VGG16, VGG19,
Inception. VGG16: VGG16 is a convolutional neural net (CNN) architecture,
which is considered to be one of the excellent vision model architectures to date. It
is 16 layers that have some weights.
- VGG19: VGG19 is a convolutional neural net (CNN) architecture that is 19 layers
deep.
- Inception: It is a convolutional neural network for assisting in image analysis and
object detection.

### 4. TRAINING THE MODEL
The model shall be trained using 80% of images. The metrics used are Accuracy.
- Accuracy: It is one of the metrics used for evaluating the classification models.
Accuracy is the ratio of predictions the model got right to the total number of
predictions.

### 5. TESTING THE MODEL
The model shall be tested by using the remaining 20% of images to validate the
accuracy and ensure that the model correctly classifies with minimal error.
