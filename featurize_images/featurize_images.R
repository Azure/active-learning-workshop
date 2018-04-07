## This script featurizes a large number of images, on Azure Batch
#
# * Subset of the 'faces' dataset
# * The wood knots dataset
# * The Caltech256 dataset

##########################################################################################
#### Environment setup

install.packages(c("devtools", "curl", "httr", "imager"));
install.packages(c("dplyr", "tidyr", "ggplot2", "magrittr", "digest", "RCurl", "foreach"));

devtools::install_github("Azure/rAzureBatch")
devtools::install_github("Azure/doAzureParallel")

# CRAN prerequisities for AzureSMR
install.packages(c('assertthat', 'XML', 'base64enc', 'shiny', 'miniUI', 'DT', 'lubridate'));
devtools::install_github("Microsoft/AzureSMR");

# Participants: Add Microsoft goodness to path on the DSVMs.
#
# After installing all other packages, execute in the DSVM terminal: 
# 
# sudo systemctl start rstudio-server
# sudo echo "r-libs-user=/data/mlserver/9.2.1/libraries/RServer" >>/etc/rstudio/rsession.conf
# 
# This adds the RServer libraries to path for all sessions, including those spawned for parallel local.
#
# .libPaths( c( "/data/mlserver/9.2.1/libraries/RServer", .libPaths()))
# library(RevoScaleR)
# library(MicrosoftML)

# Now follow azureparallel_setup to start a Batch cluster.


##########################################################################################
## Data locations, names, keys...

storageAccount = "storage4tomasbatch";
source("secrets.R") # get the account key
container = "tutorial";
BLOB_URL_BASE = paste0("https://", storageAccount, ".blob.core.windows.net/", container, '/');

CALTECH_FEATURIZED_DATA='Caltech.Rds'
KNOTS_FEATURIZED_DATA='knots.Rds'
FACES_SMALL_FEATURIZED_DATA='faces_small.Rds'

##########################################################################################
## list blob contents in a single directory and parse it into constituent parts

library(AzureSMR)   # AzureSMR is only needed on the head node - to list files in blob

get_blob_info <- function(storageAccount, storageKey, container, prefix) {
    marker = NULL;
    blob_info = NULL;
    repeat {
        info <- azureListStorageBlobs(NULL,
                                   storageAccount = storageAccount,
                                   storageKey = storageKey,
                                   container = container,
                                   marker = marker,
                                   prefix = prefix)

        if (is.null(blob_info)) {
            blob_info = info;
        } else {
            blob_info = rbind(blob_info, info)
        }

        marker <- attr(info, 'marker');
        print(paste0("Have ", nrow(blob_info), " blobs"));
        if (marker == "") {
            break
        } else {
            print("Still more blobs to get")
        }
    }
    # end blob directory read loop

    # preprocess the file names into urls and class (person) names
    blob_info$url <- paste(BLOB_URL_BASE, sep = '', blob_info$name)
    blob_info$fname <- sapply(strsplit(blob_info$name, '/'), function(l) { l[length(l)] })
    blob_info$bname <- sapply(strsplit(blob_info$fname, ".", fixed = TRUE), function(l) l[1])
    blob_info$pname <- sapply(strsplit(blob_info$fname, "_", fixed = TRUE),
                          function(l) paste(l[1:(length(l) - 1)], collapse = " "))

    return(blob_info);
}
# end get_blob_info


##########################################################################################
# Parallel kernel for featurization
parallel_kernel <- function(blob_info) {
  
  # this ensures we will find the RServer libs on the DSVM
  RSERVER_LIBS="/data/mlserver/9.2.1/libraries/RServer";
  if (!(RSERVER_LIBS %in% .libPaths() ) ) {
   .libPaths( c(RSERVER_LIBS, .libPaths()))
  }

  library(MicrosoftML)
  library(utils)
  
  # get the images from blob and do them locally
  DATA_DIR <- file.path(getwd(), 'localdata');
  if(!dir.exists(DATA_DIR)) dir.create(DATA_DIR);
  
  # do this in paralell, too
  for (i in 1:nrow(blob_info)) {
    targetfile <- file.path(DATA_DIR, blob_info$fname[[i]]);
    if (!file.exists(targetfile)) {
      download.file(blob_info$url[[i]], destfile = targetfile, mode="wb")
    }
  }
  
  blob_info$localname <- paste(DATA_DIR, sep='/', blob_info$fname);
  
  # featurize using the rx-functions
  image_features <- rxFeaturize(data = blob_info,
                                mlTransforms = list(loadImage(vars = list(Image = "localname")),
                                                    resizeImage(vars = list(ResImage = "Image"),
                                                                width = 224, height = 224),
                                                    extractPixels(vars = list(Features = "ResImage"))
                                                    , featurizeImage(var = "Features", dnnModel = 'resnet18')
                                                    ),
                                mlTransformVars = c("localname"))
  
  image_features$url <- blob_info$url;
  return(image_features)
}


##########################################################################################
## Splitting the dataset for parallel run

# Returns a list of data frame shards (smaller data frames), each enclosed in one-element list
#
# Note: Parallel execution expects a list of argument vectors. 
#       Since parallel_kernel has 1 argument, each argument vector has length one. 
#       That is the "extra" layer of list.
shardDataFrame <- function(df, shardcount) {
  N <- dim(df)[1]  
  batch_size <- ceiling(N/shardcount);
  
  shards <- lapply(1:shardcount, function(i) {
    fromRow = (i-1)*batch_size+1;
    toRow = min(i*batch_size, N);  
    return(list(df[fromRow:toRow,])); # extra list()
  } )
}


##########################################################################################
## Featurize the faces dataset

# Get the list of images
blob_info <- get_blob_info(storageAccount, storageKey, container, prefix = "faces_small");

# Version 0: test that things work locally
start_time <- Sys.time()
output <- parallel_kernel(blob_info);
end_time <- Sys.time()
print(paste0("Local ran for ", round(as.numeric(end_time - start_time, units="secs")), " seconds"))



SLOTS=4 # parallelism level. We have 4 codes. 
shards <- shardDataFrame(blob_info, SLOTS)      # create a list of dataframes, a uniform partition of blob_info

# Option 1: run the featurization locally on singlecore
rxSetComputeContext(RxLocalSeq());
start_time <- Sys.time()
outputs <- rxExec(FUN=parallel_kernel, elemArgs=shards) # list of task outputs
end_time <- Sys.time()
print(paste0("Local sequential ran for ", round(as.numeric(end_time - start_time, units="secs")), " seconds"))
## The sharding adds about 3 seconds to serial execution



# Option 2: run the featurization locally on multicore
rxOptions(numCoresToUse=SLOTS);
rxSetComputeContext(RxLocalParallel());
start_time <- Sys.time()
outputs <- rxExec(FUN=parallel_kernel, elemArgs=shards)
end_time <- Sys.time()
print(paste0("Local parallel ran for ", round(as.numeric(end_time - start_time, units="secs")), " seconds"))
## Actually slower. There is no benefit of parallelizing a multithreaded workload here, just overhead.


# Option 3: Run the featurization on Azure Batch
start_time <- Sys.time()
outputs <- foreach(shard=shards) %dopar% {     # %dopar% invokes parallel backend (registered cluster)
  parallel_kernel(shard[[1]]) # shards are argument 1-tuples, kernel takes the element of the 1-tuple
}
end_time <- Sys.time()
print(paste0("Azure parallel ran for ", round(as.numeric(end_time - start_time, units="secs")), " seconds"))



# Option 4: Run the featurization in a Spark cluster
#
# mySparkCluster <- rxSparkConnect()
#
# run the featurization on the cluster
# rxSetComputeContext(mySparkCluster);
# start_time <- Sys.time()
# outputs <- rxExec(FUN=parallel_kernel, elemArgs=shards)
# end_time <- Sys.time()
# print(paste0("Spark ran for ", round(as.numeric(end_time - start_time, units="secs")), " seconds"))

# the output is a list of length SLOTS, collect back into one dataframe
faces_small_df <- Reduce(rbind, outputs)


#####################################################################################
# makes sense?

library(tidyr)
library(ggplot2)
library(magrittr)
faces_small_df$pname <- blob_info$pname
faces_small_df$bname <- blob_info$bname
features <- faces_small_df %>% gather(featname, featval, -pname)    # plot features by file
plottable <- features[startsWith(features$featname, 'Feature'),];
plottable$featval <- type.convert(plottable$featval);                       # make numeric again
(
  p <- ggplot(plottable, aes(featname, pname)) + 
    geom_tile(aes(fill = featval), colour = "white") +
    scale_fill_gradient(low = "white", high = "steelblue")
)

##########################################################################################
##########################################################################################
## Featurize the Caltech dataset

## the caltech dataset has a two-level directory structure, so the parsing 
## will be a bit different from get_blob_info
get_caltech_info <- function(storageAccount, storageKey, container, prefix) {
  marker = NULL;
  blob_info = NULL;
  repeat {
    info <- azureListStorageBlobs(NULL,
                                  storageAccount = storageAccount,
                                  storageKey = storageKey,
                                  container = container,
                                  marker = marker,
                                  prefix = prefix)
    
    if (is.null(blob_info)) {
      blob_info = info;
    } else {
      blob_info = rbind(blob_info, info)
    }
    
    marker <- attr(info, 'marker');
    print(paste0("Have ", nrow(blob_info), " blobs"));
    if (marker == "") {
      break
    } else {
      print("Still more blobs to get")
    }
  }
  # end blob directory read loop
  
  # preprocess the file names into urls and class (person) names
  blob_info$url <- paste(BLOB_URL_BASE, sep = '', blob_info$name)
  blob_info$fname <- sapply(strsplit(blob_info$name, '/'), function(l) { l[length(l)] })
  blob_info$bname <- sapply(strsplit(blob_info$fname, ".", fixed = TRUE), function(l) l[[1]])
  blob_info$clsid <- sapply(strsplit(blob_info$name, '/'), function(l) { l[2] })
  blob_info$cname <- sapply(strsplit(blob_info$clsid, ".", fixed = TRUE), 
                            function(l) { l[2] }
                            # the second part of the "001.ak47" string, which is the second directory
                            );
  
  return(blob_info);
}


### load or make featurized data, on Azure Batch
if( file.exists(CALTECH_FEATURIZED_DATA)){
  
  caltech_df <- readRDS(CALTECH_FEATURIZED_DATA)
  
} else {

  caltech_info <- get_caltech_info(storageAccount, storageKey, container, prefix = "256_ObjectCategories");
  
  SLOTS = 96
  caltech_shards <- shardDataFrame(caltech_info, SLOTS)
  start_time <- Sys.time()
  outputs <- foreach(shard=caltech_shards) %dopar% {     # %dopar% invokes parallel backend (registered cluster)
    parallel_kernel(shard[[1]]) # shards are argument 1-tuples, kernel takes the element of the 1-tuple
  }
  end_time <- Sys.time()
  print(paste0("Azure parallel ran for ", round(as.numeric(end_time - start_time, units="secs")), " seconds"))
  caltech_df <- Reduce(rbind, outputs)
  caltech_df$cname <- caltech_info$cname;
 
  saveRDS(caltech_df, CALTECH_FEATURIZED_DATA);
}


##########################################################################################
## Wood knots dataset

### load or make featurized data, on local multicore
if( file.exists(KNOTS_FEATURIZED_DATA)){
  
  knots_df <- readRDS(KNOTS_FEATURIZED_DATA)
  
} else {
  
  knots_info <- get_blob_info(storageAccount, storageKey, container, prefix = "knot_images_png");

  # Just use my 4 local cores
  SLOTS=4 
  rxOptions(numCoresToUse=SLOTS);
  rxSetComputeContext(RxLocalSeq());

  knots_shards <- shardDataFrame(knots_info, SLOTS);      # create a list of dataframes, a uniform partition of blob_info
  start_time <- Sys.time()
  outputs <- rxExec(FUN=parallel_kernel, elemArgs=knots_shards);
  end_time <- Sys.time()
  print(paste0("Local parallel ran for ", round(as.numeric(end_time - start_time, units="secs")), " seconds"))
  
  knots_df <- Reduce(rbind, outputs);
  
  saveRDS(knots_df, KNOTS_FEATURIZED_DATA);
  
}


##########################################################################################
## Featurization maps disparate images into the same space

dim(caltech_df)
dim(knots_df)

# What does this woodknot look like?
# Find the thing in the Caltech dataset that is the closest in L1 sense and show it.
# Returns the row from dataset_df that is closest to single_row_df
# in the L1 sense. Columns must match in dataset_df and single_row_df.
#
# This is a table scan. I don't know anything about clever indexing 
# and table scans sell cloud compute.
find_L1_closest <- function (dataset_df, single_row_df) {
  
  numdf <- dplyr::select_if(dataset_df,is.numeric);
  single <- dplyr::select_if(single_row_df,is.numeric);
  
  N = nrow(numdf);
  diffs <- sweep(numdf, 2, as.numeric(single), "-");
  numdf$l1 <- rowSums(abs(diffs));
  closest <- which(numdf$l1 == min(numdf$l1));
  
  return( dataset_df[closest,])
  
}

showurl <- function(url) {
  if (is.list(url)) {
    if (length(url) > 1) {
      print("Warning: only one url will be shown")
    }
    oneurl <- url[[1]]
  } else {
    oneurl <- url;
  }
  
  urlpieces <- strsplit(oneurl, '/')[[1]];
  
  DATA_DIR <- file.path(getwd(), 'localdata');
  if(!dir.exists(DATA_DIR)) dir.create(DATA_DIR);
  
  targetfile <- file.path(DATA_DIR, urlpieces[[length(urlpieces)]]);
  
  download.file(url, destfile = targetfile, mode="wb");
  i <- load.image(targetfile)
  cls <- urlpieces[length(urlpieces) - 1]
  plot(i, main=cls)
}


library(imager)
library(dplyr)

# show a woodknot
WHICH_KNOT=1
showurl(knots_df[WHICH_KNOT,"url"])

lookslike <- find_L1_closest(caltech_df, knots_df[WHICH_KNOT, ])
showurl(lookslike$url)


WHICH_FACE=1
showurl(faces_small_df[WHICH_FACE,"url"])

lookslike <- find_L1_closest(caltech_df, faces_small_df[WHICH_FACE, ])
showurl(lookslike$url)

# OMG that lookup was slow.... Hey, I have a cluster! :)
