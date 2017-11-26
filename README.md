![alt text](https://github.com/cyrilzakka/HealthKit-ML/blob/master/banner.png)

# Introduction
HealthKit ML aims to bring artifical intelligence to healthcare, one neural network at a time. With the recent advances in machine learning, or more specifically the successes garnered by deep learning algorithms in tasks such as computer vision and natural language processing (NLP), translating those achievements over to the medical field seems like a natural step in the right direction - with the intended hope of increasing the accuracy of diagnoses and the accessibility of healthcare overall. 

# Requirements
- Keras (TensorFlow backend)
- Python 3.6.0 or later
- numpy/pandas/matplotlib
- openCV

# Table of Contents
- [Computer Vision](#computer-vision)
  1. [TB-Net: Tuberculosis Detection from Chest X-Rays](#tb-net-tuberculosis-detection-in-chest-x-rays)
- [Natural Language Processing](#natural-language-processing)
- [Model Integration](#model-integration)
  1. [iOS and Core ML](#ios-and-core-ml)

# Computer Vision
### TB-Net: Tuberculosis Detection in Chest X-Rays 
#### Background
Tuberculosis (TB) is an infectious disease that mainly affects the lungs. It is caused by the _Mycobacterium tuberculosis_  bacterium (MTB), and spreads through airborne droplets from an infected person. While most infections do not have symptoms (latent tuberculosis), about 10% of cases progress to active TB which, if left untreated, kills about half of those infected. Symptoms include chronic coughing, bloody sputum, fever, night sweats, and weight loss.

Despite the discovery of antibiotic drugs in the 1940s, TB has seen a resurgence due to increasing rates of multiple drug-resistant tuberculosis (MDR-TB) and HIV/AIDS. In fact, one-third of the world's population is thought to be infected with TB, with around 1.37 million reported deaths per year.

### Purpose
With the drought in radiologists currently being experienced by developing countries in the world, diagnosing various diseases for proper treatment has become quite the challenge. Can neural networks be leveraged to help narrow the gap between the supply and demand of radiologists in the diagnosis of TB?

### Data & Training
Datasets containing chest X-rays of TB-infected patients and otherwise healthy individuals were downloaded from the [National Institute of Health (NIH)](https://ceb.nlm.nih.gov/repositories/tuberculosis-chest-x-ray-image-data-sets/) and split into 8:1:1 for training, validation and test sets respectively - images from both datasets were rescaled `1./255` and resized to `224*224` and distributed equally among the training, validation and test sets.

Data augmentation was used limiting the `zoom`, `shearing`, `height/width shift` ranges to `0.2`, `rotation angle` to `40°` and `horizontal flipping` set to `True`.

Training took place at around `268s/epoch` for `30 epochs` on a single Tesla K80 GPU using a 15-layer convolutional neural network with `dropout` at  `(0.5)` to minimize overfitting.

### Results and Limitations
| Metric        | Value         |
|:--------------|--------------:|
| Test Accuracy |    86.7%      |
| ROC AUC Score |     0.89      |

# Natural Language Processing
# Model Integration
### iOS and Core ML 
