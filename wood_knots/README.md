# Image classification workshop using featurization and active learning

## Setup

* Create a Data Science Virtual Machine - Windows 2016 using VM size DS13_v2 (8 vcpus)
* Launch “git bash” on the Data Science Virtual Machine
* In git bash, run this command: git clone https://github.com/Azure/MLADS2017ML.git
* Open 1_woodknots_active_learning_workshop.Rmd in RStudio and click “Knit”.
* Optional: Set up an Azure Batch account for use with doAzureParallel, as described here: https://github.com/Azure/doAzureParallel
* Open azureparallel_setup.R in RStudio and optionally run the lines to launch one or both Azure Batch clusters.
* Open azureparallel_tutorial.R and run the code, choosing the small or large data sets.


## Image Featurization

The use case (labeling knots in lumber) and concepts of image featurization are described in our blog post entitled [Featurizing Images: the shallow end of deep learning](http://blog.revolutionanalytics.com/2017/09/wood-knots.html). Briefly, a pre-trained DNN image classification model is used to generate features for a set of images, which are then used to train a custom classifier. This is the simplest form of transfer learning, where the values leading into the last layer of the network are used as features, and we do not use backpropagation on the original DNN model.

These images are from the [University of Oulu](http://www.ee.oulu.fi/~olli/Projects/Lumber.Grading.html), Finland. The [labelled images](http://www.ee.oulu.fi/research/imag/knots/KNOTS) were saved as individual knot images by the original authors, and we segmented the "unlabelled" images by hand using [LabelImg](https://github.com/tzutalin/labelImg). We have converted all of the individual knot images to PNG format, and you can download zip files containing PNG versions of the [labelled images](https://isvdemostorageaccount.blob.core.windows.net/wood-knots/labelled_knot_images_png.zip) and the [segmented unlabelled images](https://isvdemostorageaccount.blob.core.windows.net/wood-knots/unlabelled_cropped_png.zip) from Azure blob storage.

## Active Learning

[Active learning](https://en.wikipedia.org/wiki/Active_learning) helps us address the common situation where we have large amounts of data, but labeling this data is expensive. By using a preliminary model to select the cases that are likely to be most useful for improving the model, and iterating through several cycles of model training and case selection, we can often build a model using a much smaller training set (thus requiring less labeling effort) than we would otherwise need. Companies like [CrowdFlower](https://www.crowdflower.com/) and services like the [Azure Custom Vision service](https://azure.microsoft.com/en-us/services/cognitive-services/custom-vision-service/) make use of active learning.

### Image Labeling website

Our [label collection website](https://woodknotlabeler.azurewebsites.net) has instructions for how to recognize the different classes of knots, a page where you can practice, and a page where workshop participants can enter their labels for the images chosen for the first round of active learning. The code for this web app is in its own [gitHub repo](https://github.com/jichang1/woodknotlabeller).

## Outline

0. Welcome and setup (10 min)
	* provisioning compute resources

1. Introduction (20 min)
    - Deep learning as feature development + classification/regression
    - Use case: wood knot classification for grading lumber
        - build a classifier to make distinctions the original model was not trained for
        - transfer of low-level features (edges, etc.)
        - labeled training set: three classes of knots
    - Relationship to transfer learning in general
		- more data lets you tune weights further up the stack of layers

**Activity 1:** Featurizing Images at Scale (20 min)

2. Active Learning: Background (10 min) 
	* data is often easier to come by than expert labels
	* use the preliminary model for triage of unlabeled data
		- What is the model good at? What needs work (e.g., more training data)?
		- How much of the unlabeled data can we eliminate as already identifiable?
	* better model -> better triage -> better selection of cases to label -> better model -> ...
	* Companies like CrowdFlower and services like the Custom Vision Service use active learning.

3. Active Learning: First Round (10 min)
	(Walk through first part of Active Learning Workshop RMD file)
	* Build and evaluate model
		- Build initial classification model
			- Handling wide data: more columns than rows
			- ordinary least squares regression is underdetermined
			- regularization (or sampling and ensembling) to the rescue
			- this is not specific to featurization, only to the number of features
		- performance
			- multiclass classifier: confusion matrix
		    - ROC curves for one class at a time
		- hyperparameters (we're using fixed values of L1 and L2 penalties, but they are important)
	* Select the most useful images to label
		- worst best score
		- information entropy

**Activity 2:** Label selected images (15 min)

4. Active Learning: Iterative Improvement (15 min) 
	(Walk through rest of Active Learning Workshop RMD file)

**Demo:** Evaluating Participant Performance (5 min) 



## Outline

0. Welcome and setup (10 min)
	* provisioning compute resources

1. Introduction (20 min)
    - Deep learning as feature development + classification/regression
    - Use case: wood knot classification for grading lumber
        - build a classifier to make distinctions the original model was not trained for
        - transfer of low-level features (edges, etc.)
        - labeled training set: three classes of knots
    - Relationship to transfer learning in general
		- more data lets you tune weights further up the stack of layers

**Activity 1:** Featurizing Images at Scale (20 min)

2. Active Learning: Background (10 min) 
	* data is often easier to come by than expert labels
	* use the preliminary model for triage of unlabeled data
		- What is the model good at? What needs work (e.g., more training data)?
		- How much of the unlabeled data can we eliminate as already identifiable?
	* better model -> better triage -> better selection of cases to label -> better model -> ...
	* Companies like CrowdFlower and services like the Custom Vision Service use active learning.

3. Active Learning: First Round (10 min)
	(Walk through first part of Active Learning Workshop RMD file)
	* Build and evaluate model
		- Build initial classification model
			- Handling wide data: more columns than rows
			- ordinary least squares regression is underdetermined
			- regularization (or sampling and ensembling) to the rescue
			- this is not specific to featurization, only to the number of features
		- performance
			- multiclass classifier: confusion matrix
		    - ROC curves for one class at a time
		- hyperparameters (we're using fixed values of L1 and L2 penalties, but they are important)
	* Select the most useful images to label
		- worst best score
		- information entropy

**Activity 2:** Label selected images (15 min)

4. Active Learning: Iterative Improvement (15 min) 
	(Walk through rest of Active Learning Workshop RMD file)

**Demo:** Evaluating Participant Performance (5 min) 



## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
