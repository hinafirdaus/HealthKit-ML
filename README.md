![alt text](https://github.com/cyrilzakka/HealthKit-ML/blob/master/banner.png)

# Introduction
HealthKit ML aims to bring artifical intelligence to healthcare, one neural network at a time. With the recent advances in machine learning, or more specifically the successes garnered by deep learning algorithms in tasks such as computer vision and natural language processing (NLP), translating those achievements over to the medical field seems like a natural step in the right direction - with the intended hope of increasing the accuracy of diagnoses and the accessibility of healthcare overall.

**Note:** These models are only intended as proof-of-concepts and should not be used in production, clinical settings or otherwise.

# Requirements
- Keras (TensorFlow backend)
- Python 3.6.0 or later
- numpy/pandas/matplotlib
- openCV

# Table of Contents
- [Computer Vision](#computer-vision)
  1. [TB-Net: Tuberculosis Detection from Chest X-Rays](#tb-net-tuberculosis-detection-in-chest-x-rays)
- [Natural Language Processing](#natural-language-processing)
- [Sequence Models](#sequence-models)
  1. [Cardya Heart Rate Monitor: A Case Study](#cardya-heart-rate-monitor-a-case-study)
- [Model Integration](#model-integration)
  1. [iOS and Core ML](#ios-and-core-ml)

# Computer Vision
### TB-Net: Tuberculosis Detection in Chest X-Rays 
#### Background
Tuberculosis (TB) is an infectious disease that mainly affects the lungs. It is caused by the _Mycobacterium tuberculosis_  bacterium (MTB), and spreads through airborne droplets from an infected person. While most infections do not have symptoms (latent tuberculosis), about 10% of cases progress to active TB which, if left untreated, kills about half of those infected<sup>[1](#references)</sup>. Symptoms include chronic coughing, bloody sputum, fever, night sweats, and weight loss.

Despite the discovery of antibiotic drugs in the 1940s, TB has seen a resurgence due to increasing rates of multiple drug-resistant tuberculosis (MDR-TB) and HIV/AIDS<sup>[2](#references)</sup>. In fact, one-third of the world's population is thought to be infected with TB, with around 1.37 million reported deaths per year<sup>[3](#references)</sup>.

#### Purpose
With the drought in radiologists currently being experienced by developing countries in the world<sup>[4](#references)</sup>, diagnosing various diseases for proper treatment has become quite the challenge. Can neural networks be leveraged to help narrow the gap between the supply and demand of radiologists in the diagnosis of TB?

#### Data & Training
Datasets containing chest X-rays of TB-infected patients and otherwise healthy individuals were downloaded from the [National Institute of Health (NIH)](https://ceb.nlm.nih.gov/repositories/tuberculosis-chest-x-ray-image-data-sets/) and split into 8:1:1 for training, validation and test sets respectively - images from both datasets were rescaled `1./255` and resized to `224*224` and distributed equally among the training, validation and test sets.

Data augmentation was used limiting the `zoom`, `shearing`, `height/width shift` ranges to `0.2`, `rotation angle` to `40°` and `horizontal flipping` set to `True`.

Training took place at around `268s/epoch` for `30 epochs` on a single Tesla K80 GPU using a 15-layer convolutional neural network with `dropout` at  `0.5` to minimize overfitting.

#### Results and Limitations

| Metric                 | Value         |
|:-----------------------|--------------:|
| Test Accuracy          |     0.87      |
| Test Average Precision |     0.94      |
| ROC AUC Score          |     0.89      |

Despite obtaining relatively high metrics on this task, the original dataset should be increased with additional data in order to further improve the generalizability of this model. Further work should look into automatically identifying the regions of concern either using bounding-boxes (by supplying appropriately labelled data) or using [class-activation mapping (CAM)](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf), as well as being able to differentiate between different diseases in a single chest x-ray such as pneumonia or nodules.

#### References
1. World Health Organization. Tuberculosis Fact Sheet. http://www.who.int/mediacentre/factsheets/fs104/en/. Accessed November 26, 2017.
2. University of Maryland Medical Center. Tuberculosis. http://www.umm.edu/health/medical/altmed/condition/tuberculosis. Accessed November 26, 2017.
3. World Health Organization. Trade, foreign policy, diplomacy and health: Tuberculosis Control http://www.who.int/trade/distance_learning/gpgh/gpgh3/en/index4.html. Accessed November 26, 2017.
4. Radiological Society of North America. Developing Countries in Dire Need of Radiology Training. http://www.rsna.org/NewsDetail.aspx?id=12532. Accessed November 26, 2017.

# Natural Language Processing
Coming soon!
# Sequence Models
### Cardya Heart Rate Monitor: A Case Study
#### Background
The human body is a complex entity: problems rarely occur without any early warning signs, and learning to detect them prematurely might be the difference between life and death for many. With the growing tide of wearables such as the Apple Watch and FitBit, monitoring patient lifestyle and health parameters outside of the clinic becomes an all too easy task. What if such devices could be used to detect and identify heart disease?

Cardya is an iOS application designed to collect and aggregate annotated heart data from its users, in an effort to develop models capable of detecting and identifying heart conditions such as atrial fibrillation and ventricular tachycardias. Its development can be divided into 2 distinct phases:

1. _Data Collection and Annotation_: Heart rate data, previously diagnosed heart conditions and various health parameters (i.e. height, weight, age, etc.) are collected from the user and stored on our servers.
2. _Model Development and Testing_: Data is cleaned, triaged and fed into a neural network in the hopes of developing a model capable of detecting various heart conditions with accuracies upward of 0.90.

#### Purpose
How effective is crowdsourcing in the context of healthcare? Can wearables be used to detect potentially lethal heart conditions?

### Screenshots

<img src="https://user-images.githubusercontent.com/1841186/35160683-e3d40abe-fd0b-11e7-8624-835f020483d8.PNG" width="270">   <img src="https://user-images.githubusercontent.com/1841186/35160686-e3fd6d1e-fd0b-11e7-899f-43115e0d7f55.PNG" width="270">   <img src="https://user-images.githubusercontent.com/1841186/35160687-e40d5d32-fd0b-11e7-83ae-83ad56197331.PNG" width="270">

# Model Integration
### iOS and Core ML 
#### Background
Properly training a neural network is one thing, and trying to get a model to make millions of predictions on-device in real-time is a different task entirely. Thankfully with the advent of iOS 11, Apple introduced [CoreML](https://developer.apple.com/documentation/coreml), a foundational framework it describes as 'deliver[ing] blazingly fast performance with easy integration of machine learning models' on iOS devices. 

#### Purpose
In countries with insufficient and unreliable access to the internet, making predictions offline and in real-time can come quite in handy, especially when your model can be downloaded, installed and updated just as easily as any other application from the AppStore.

#### Procedure
##### CoreMLTools
In order to convert your model to something iOS can use, you'll have first to install `coremltools`. Here's how to quickly get setup in a `virtualenv` environment. 

**Note:** As of the time of writing, `coremltools` only works with Python 2.7 and raises errors upon installation with Python 3.6.0 +. As such, a `virtualenv` with Python 2.7 is used.

Creating the `virtualenv` with Python 2.7: 
```bash
pip install virtualenv
virtualenv --python=/usr/bin/python2.7 <DIR>
source <DIR>/bin/activate
```

Installing coremltools and its dependencies:
```bash
pip install -U coremltools
pip install tensorflow
pip install keras
pip install h5py
```

Converting your model:
```python
## Inside of the python interpreter
import coremltools
coreml_model = coremltools.converters.keras.convert('tb_net.h5')
coreml_model.author = 'Cyril Zakka'
coreml_model.license = 'MIT'
coreml_model.short_description = 'Predicts the probability of TB in a chest X-ray.'
coreml_model.save('tb_net.mlmodel')
```
##### Xcode and Vision Framework
Now that you've obtained a copy of your model in a format Core ML can understand, drag it into Xcode and you'll find an option to automatically generate a Swift model class: doing so will create a class for your inputs and outputs, as well as a main class for your model along with two methods for outputting predictions. You can read more about incorporating your model into Xcode with the Vision framework [here](https://www.raywenderlich.com/164213/coreml-and-vision-machine-learning-in-ios-11-tutorial).
