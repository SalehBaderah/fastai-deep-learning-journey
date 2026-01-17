# Chapter 2: From Model to Production

## Starting Your Project
Working on a real project gives you real experience in building and using models. When selecting a project, the most important consideration is <span style="color: red;">Data Availability</span>.

## Computer Vision
Deep learning has achieved great success in various computer vision tasks:

* **Object Recognition:** Computers can recognize items in an image.
* **Object Detection:** Computers can recognize where objects are in an image, highlight their locations, and name each found object.
    * **Segmentation:** A specialized task where every pixel is categorized based on the kind of object it is part of.

### Data Limitations
Deep learning algorithms are generally not good at recognizing images that are significantly different in structure from those used to train the model.
* *Example:* If there were no black-and-white images in the training data, the model may do poorly on black-and-white images in production.

### Data Augmentation
**Data Augmentation** refers to synthetically expanding a dataset by creating modified copies of existing data (such as rotating or flipping images). This technique helps to:
1.  Train machine learning models better.
2.  Prevent overfitting.
3.  Improve the ability to generalize to new, unseen data (especially when the original dataset is small).
---
# The Drivetrain Approach
The **Drivetrain Approach** is a strategic framework for building data products using AI. Its main goal is to ensure that data is used to produce **actionable results** rather than just generating predictions.

### Core Idea
The basic idea is to start with considering your objective, then think about what actions you can take to meet that objective and what data you have (or can acquire) that can help, and then build a model that you can use to determine the best actions to take to get the best results in terms of your objective

<div align="center">
  <img src="https://github.com/SalehBaderah/fastai-deep-learning-journey/blob/main/Images/Screenshot%202026-01-15%20150659.png" width="800" alt="Drivetrain">
</div>

### The 4 Steps of the Drivetrain Approach

1.  **Define the Objective**
    * *What is the specific result we want to achieve?*
    * **Example:** For an e-commerce store, the goal isn't just to "predict sales" (which is passive), but to "maximize net profit" (which is actionable).

2.  **Define the Levers**
    * *What are the inputs we can control or change to influence the objective?*
    * **Example:** The price of a product, the placement of an ad, or the recommendation shown to a user.

3.  **Collect the Data**
    * *What data do we need to link the "Levers" to the "Objective"?*
    * You need data that shows how changing the levers affects the outcome. If you don't have this data, you may need to run experiments to generate it.

4.  **Build the Model**
    * The models in this approach don't just predict the future; they predict how pulling different levers will impact the objective.
    * **Example:** Build a model that predicts: *"If we set the price to 100, we sell 50 units. If we set it to 90, we sell 80 units."*
    * **Action:** You then use these predictions to pick the exact price (lever) that results in the highest total profit (objective).

---
# Gathering Data
For many types of projects, you might find the data you need online (e.g., Kaggle). However, it is crucial to remember that **models can reflect only the data used to train them**. If the data is biased, the model will be biased. Therefore, we must carefully consider the types and sources of our data.
# From Data to DataLoaders
- `DataLoader`: is a class that prepare the data we pass to it by making a training set and validation set.
  
Later in the book we will cover :
  - What kinds of data we are working with?
  - How to get the list of items
  - How to label these items
  - How to create the validation set

### Independent & Dependent variables
 **Independent Variable ($x$):** The data we use to make predictions (e.g., the image).
 **Dependent Variable ($y$):** The target or label we want to predict (e.g., the type of bear).


### Image Resizing
**`Resize`:** Using `Resize(128)` crops the images to fit a square shape. This can result in losing important details if the object is cut out.

**`RandomResizedCrop`:** This is the preferred solution for training. It randomly selects a part of the image and crops it. Since we select a different part of the image in each epoch, the model learns to focus on and recognize different features of the object.

**Code Example on RandomResizedCrop:**
```python
bears = bears.new(item_tfms=RandomResizedCrop(128, min_scale=0.3))
dls = bears.dataloaders(path)
dls.train.show_batch(max_n=4, nrows=1, unique=True)
```
<div align="center">
  <img src="https://github.com/SalehBaderah/fastai-deep-learning-journey/blob/main/Images/Screenshot%202026-01-16%20224331.png" width="800" alt="Drivetrain">
</div>

### Data Augmentation
Artificially expands a dataset by creating modified copies of existing data.
EX: 
- Rotation
- Flipping
- Perspective warping
- Brightness changes
- Contrast changes

**Code Example:**
```python
bears = bears.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
dls = bears.dataloaders(path)
dls.train.show_batch(max_n=8, nrows=2, unique=True)
```

<div align="center">
  <img src="https://github.com/SalehBaderah/fastai-deep-learning-journey/blob/main/Images/Screenshot%202026-01-16%20224902.png" width="800" alt="Drivetrain">
</div>


# Training the model
After we prepare the data we create a learner and fine-tune it.

```python
bears = bears.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
dls = bears.dataloaders(path)

learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)
```
## Interpretation: **Confusion Matrix**
To understand where the model is making mistakes
```python
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
```
<div align="center">
  <img src="https://github.com/SalehBaderah/fastai-deep-learning-journey/blob/main/Images/Screenshot%202026-01-16%20230228.png" width="500" alt="Drivetrain">
</div>


- **Rows**:Represent the Actual labels (the true types of bears in our dataset).
- **Columns**:Represent the Predicted labels (what the model thought the images were)
- **Diagonal**:Shows images classified correctly
- **Off-diagonal**:Shows images classified incorrectly

This helps us point exactly where errors are occurring. We can determine if it is:
1.   A Dataset Problem: e.g., images that aren't bears at all, or are labeled incorrectly.
2.   A Model Problem: e.g., the model fails on images with unusual lighting or weird angles

## Analyzing loss
To distinguish between these problems, we can sort images by their Loss. Loss is a number that indicates how "wrong" the model is. The loss is higher if the model is incorrect, and especially high if the model is confident about its incorrect answer.

We use `plot_top_losses` to see the images with the highest loss. Each image is labeled with:
1. **prediction**: What the model guessed
2. **Target: The** actual correct label.
3. **Loss**: The calculated error value.
4. **Probability**: The model's confidence level (from 0 to 1).
```python 
interp.plot_top_losses(5, nrows=1)
```
<div align="center">
  <img src="https://github.com/SalehBaderah/fastai-deep-learning-journey/blob/main/Images/Screenshot%202026-01-16%20231652.png" width="800" alt="Drivetrain">
</div>
# Turning Your Model into an Online Application
---
### Using the Model for Inference
After training a model to perform a certain task, we need to save it and then copy it to a server to use in production.

A deep learning model generally consists of:
1.  **The Architecture:** The structure of the layers.
2.  **The Trained Parameters:** The weights learned during training.
3.  **The Data Transformations:** The definition of how to process input data (e.g., resizing, cropping).

The easiest way to save a model in fastai is to use the `export` method. This saves all the necessary components (architecture, parameters, and transform definitions) into a single file.
```python
learn.export() # fastai will save a file called 'export.pkl'

path = Path()
path.ls(file_exts='.pkl') # Check that the file exists
```
## Inference
When we use a model to get **predictions** rather than for training, we call this process Inference.
To load the model for inference, we use `load_learner`

```python
learn_inf = load_learner(path/'export.pkl') # Load the model
learn_inf.predict('images/grizzly.jpg')# make predictions for one image

# OUTPUT (Returned values from .predict)

# ('grizzly', tensor(1), tensor([9.0767e-06, 9.9999e-01, 1.5748e-07])) [predicted_class,index,prob,]
```
1. Predicted Class: The label of the prediction (e.g., 'grizzly')
2. Label Index: The internal index of the category (e.g., tensor(1)).
3. Probabilities: A tensor containing the probability for each category
the model predicts the image is a grizzly with a probability of ~1.00 (99.99%)
---
# Issues with Deep learning model's behavior
In a Neural Network, behavior emerges from the model's attempt to match the training data, rather than being explicitly programmed or defined. This reliance on data can result in disaster if the training environment does not match the real-world environment.

### Example: The Bear Detection System
Consider the bear detection system, designed to warn campers in national parks.
If we train a model using a dataset downloaded from the `Bing Image Search API`, but deploy it on video cameras around campsites, we will face several significant problems:

* **Video vs. Images:** We are working with video data in production, but trained on static images.
* **Lighting Conditions:** Dealing with nighttime images, which likely do not appear in the "perfect" Bing search results.
* **Resolution:** Handling low-resolution security camera images compared to high-resolution web photos.
* **Latency:** Ensuring results are returned fast enough to be useful (inference speed) because it is a real-time system.
* **Unusual Poses:** Recognizing bears in positions rarely seen in online photos (e.g., from behind, partially covered by bushes, or far away from the camera).

**The Solution:** To make the system useful, we cannot rely on web scraping. We need to collect the photos, create the dataset, and do the labeling ourselves to match the production environment.


### Data problems
1.  **Out-of-Domain Data:**
    This occurs when the data the model sees in production is fundamentally different from the data it saw during training (e.g., Training on internet photos vs. Production on security camera footage).

2.  **Domain Shift:**
    This happens when the type of data the model was trained on changes over time (e.g., consumer behavior changing, or seasonal changes in images).
---

## Mitigation Strategies
Out-of-domain data and domain shift are examples of the larger problem that we can never fully anticipate every behavior of a neural network. However, we can mitigate these risks using a carefully **thought-out process** for deployment:


<div align="center">
  <img src="https://github.com/SalehBaderah/fastai-deep-learning-journey/blob/main/Images/Screenshot%202026-01-17%20173206.png" width="500" alt="Drivetrain">
</div>




**Manual Process (Human in the loop)**
    Run the deep learning model in parallel with human operators. Do not use the model to drive actions directly yet. Instead, have humans check the model's output to verify whether it makes sense.

2.  **Limit the Scope**
    Before a full rollout, conduct a controlled trial. Limit the scope geographically (e.g., one specific campsite) or by time to test the model-driven approach safely.

3.  **Gradual Expansion**
    Gradually increase the scope of your rollout.
    * **Crucial Step:** Ensure you have a robust reporting system in place.
    * **Goal:** You must be aware of any significant changes to the actions being taken compared to your manual process to catch errors early.
