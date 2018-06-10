# Active Learning Workshop - Scalable Featurization, Labelling and Experimentation Using R and Python

## Instructions

1. Provision a Windows Server 2016 Data Science Virtual Machine; the size "Standard_DS12_v2" works well:
https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.windows-data-science-vm?tab=Overview
- To connect to the Data Science Virtual Machine, use Microsoft Remote Desktop from the Microsoft Store
- If you cannot connect, update both sets of inbound port rules to open port 3389

2. Launch “git bash” on the Data Science Virtual Machine

3. In git bash, run this command: git clone https://github.com/Azure/active-learning-workshop.git

4. Open 1_wiki_detox_active_learning_workshop.Rmd in RStudio and click “Knit”.

To provision many Data Science Virtual Machines using automation, see the scripts and the README file in https://github.com/Azure/active-learning-workshop/blob/master/automation_scripts

## Abstract

Accessed via R and Python APIs, pre-trained Deep Learning models and Transfer Learning are making custom classification with large or small amounts of labeled data easily accessible to data scientists and application developers. This tutorial walks you through creating end-to-end data science solutions in R and Python on cloud-based infrastructure and consuming them in production.

## Active Learning

[Active learning](https://en.wikipedia.org/wiki/Active_learning) helps us address the common situation where we have large amounts of data, but labeling this data is expensive. By using a preliminary model to select the cases that are likely to be most useful for improving the model, and iterating through several cycles of model training and case selection, we can often build a model using a much smaller training set (thus requiring less labeling effort) than we would otherwise need. Companies like [Figure Eight (formerly CrowdFlower)](https://www.figure-eight.com/) and services like the [Azure Custom Vision service](https://azure.microsoft.com/en-us/services/cognitive-services/custom-vision-service/) make use of active learning.

## Outline:

1.	Data exploration
2.	Featurization using word embeddings
3.	Active learning from selected cases
4.  Other featurization approaches
5.  Classification
6.  ROC Curves and Utility Maximization
7.  Deployment and consumption of scoring services with Azure Machine Learning

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
