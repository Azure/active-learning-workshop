## This script sets everything up for using doAzureParallel

# Install required packages and dependencies
install.packages(c("devtools", "dplyr", "tidyr", "ggplot2", "magrittr", "digest", "RCurl", "foreach"));

devtools::install_github("Azure/rAzureBatch")
devtools::install_github("Azure/doAzureParallel")

####################################################################################################
# If you haven't yet, you should follow the instructions at https://github.com/Azure/doAzureParallel
# to set up the Azure Batch account. Note that it can take a few minutes to create a batch account. 
# If you haven't, do it now!


### These instructions are from https://github.com/Azure/doAzureParallel

library(doAzureParallel)

# 1. Generate your credential and cluster configuration files.
# setwd("C:\\Users\\tosingli\\Source\\Strata2018\\featurize_images")
# generateClusterConfig("cluster_default.json")
# generateCredentialsConfig("credentials_default.json")

# 2. Fill out your credential config and cluster config files.
# Enter your Azure Batch Account & Azure Storage keys/account-info into your credential config 
# ("credentials.json") and configure your cluster in your cluster config ("cluster.json")

# 3. Set your credentials - you need to give the R session your credentials to interact with Azure
# setwd("C:\\Users\\tosingli\\Source\\Strata2018\\featurize_images")
setCredentials("credentials.json")

# 4. Register the pool. This will create a new pool if your pool hasn't already been provisioned.
cluster <- makeCluster("cluster_docker.json")   # 2-12 nodes, elastic 

### Check your cluster in Azure Portal

# 5. Register the pool as your parallel backend - one of these!

registerDoAzureParallel(cluster)

# 6. Check that your parallel backend has been registered
getDoParWorkers()

### do work on cluster

# 7. Shut down so as not to burn through your Azure money
stopCluster(cluster)
