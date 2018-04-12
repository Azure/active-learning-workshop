# Active Learning Workshop - Scalable Featurization, Labelling and Experimentation Using R and Python

## Instructions

1. Provision a CentOS Linux Data Science Virtual Machine; the size "Standard_DS12_v2" works well:
https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.linux-data-science-vm?tab=Overview

2. Log in to JupyterHub by pointing your web browser to https://hostname:8000 (be sure to use https, not http, and replace "hostname" with the hostname or IP address of your virtual machine). Please disgregard warnings about certificate errors.

3. Open a bash terminal window in JupyterHub by clicking the New button and then clicking Terminal.

In the terminal, run these four commands:

```bash
cd ~/notebooks

git clone https://github.com/Azure/active-learning-workshop

cd active-learning-workshop

source startup.sh
```

4. You can now log in to RStudio Server at http://hostname:8787 (unlike JupyterHub, be sure to use http, not https).

5. In RStudio Server, navigate to ~/notebooks/active-learning-workshop/wood_knots/1_woodknots_active_learning_workshop.Rmd and click “Knit”

To provision many Data Science Virtual Machines using automation, see the scripts and the README file in https://github.com/Azure/active-learning-workshop/blob/master/automation_scripts

## Abstract

Accessed via R and Python APIs, pre-trained Deep Learning models and Transfer Learning are making custom Image Classification with large or small amounts of labeled data easily accessible to data scientists and application developers. This tutorial walks you through creating end-to-end data science solutions in R and Python on virtual machines, Spark environments, and cloud-based infrastructure and consuming them in production. This tutorial covers strategies and best practices for porting and interoperating between R and Python, with a novel Deep Learning use case for Image Classification as an example use case.

The tutorial materials and the scripts that are used to create the virtual machines configured as single-node Spark clusters are published in this GitHub repository, so you’ll be able to create environments identical to the ones you use in the tutorial by running the scripts  after the tutorial session completes.

## Outline:

1.	What limits the scalability of R and Python scripts?
2.	What functions and techniques can be used to overcome those limits?
3.	Hands-on, end-to-end Deep Learning-based Image Classification example in R and Python using functions that scale from single nodes to distributed computing clusters
    1.	Data exploration and wrangling
    2.	Featurization and Modeling
    3.	Deployment and Consumption
    4.	Scaling with distributed computing

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
