## This script featurizes a large number of images. 
## We assume you have run through azureparallel_setup.R and you have worker machines in your cluster.


##########################################################################################
#### Get the list of blobs to process
library(AzureSMR)

BLOB_URL_BASE = "https://storage4tomasbatch.blob.core.windows.net/tutorial/";
storageAccount = "storage4tomasbatch";
storageKey = "WpJqUKKq+8dgOGIXNlubRVrLu6vdNArNW9sE+cAGdwss1ETSb3P9ihjcSbFBQitAMs7RX/avXtGAYRORhuhHZA==";
container = "tutorial";

## list blob contents
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
  } # end blob directory read loop

  # preprocess the file names into urls and class (person) names
  blob_info$url <- paste(BLOB_URL_BASE, sep='', blob_info$name)
  blob_info$fname <- sapply(strsplit(blob_info$name, '/'), function(l) {l[2]})
  blob_info$bname <- sapply(strsplit(blob_info$fname, ".", fixed=TRUE), function(l) l[1])
  blob_info$pname <- sapply(strsplit(blob_info$fname, "_", fixed=TRUE), 
                          function(l) paste(l[1:(length(l)-1)], collapse=" "))

  return(blob_info);
} # end get_blob_info

blob_info <- get_blob_info(storageAccount, storageKey, container, prefix = "faces_small");
blob_info <- get_blob_info(storageAccount, storageKey, container, prefix = "faces_full");

##########################################################################################
# Parallel kernel for featurization
parallel_kernel <- function(blob_info) {
  
  library(MicrosoftML)
  library(utils)
  
  # get the images from blob and do them locally
  DATA_DIR <- file.path(getwd(), 'localdata');
  if(!dir.exists(DATA_DIR)) dir.create(DATA_DIR);
  
  # do this in paralell, too
  for (i in 1:nrow(blob_info)) {
    download.file(blob_info$url[[i]], destfile = file.path(DATA_DIR, blob_info$fname[[i]]))
  }
  
  blob_info$localname <- paste(DATA_DIR, sep='/', blob_info$fname);
  
  image_features <- rxFeaturize(data = blob_info,
                                mlTransforms = list(loadImage(vars = list(Image = "localname")),
                                                    resizeImage(vars = list(Features = "Image"),
                                                                width = 224, height = 224,
                                                                resizingOption = "IsoPad"),
                                                    extractPixels(vars = "Features"),
                                                    featurizeImage(var = "Features",
                                                                   dnnModel = "Resnet18")),
                                mlTransformVars = c("url"),
                                reportProgress=1)
  image_features
}


##########################################################################################
#### Run the parallel kernel locally

BATCH_SIZE = nrow(blob_info);   # do it all in one batch
NO_BATCHES = ceiling(nrow(blob_info)/BATCH_SIZE);

#### local execution
start_time <- Sys.time()
results <- foreach(i=1:NO_BATCHES ) %do% {  # %do% is the serial version
  N = nrow(blob_info);
  fromRow = (i-1)*BATCH_SIZE+1;
  toRow = min(i*BATCH_SIZE, N);
  parallel_kernel(blob_info[fromRow:toRow,])
}
end_time <- Sys.time()
print(paste0("Ran for ", as.numeric(end_time - start_time, units="secs"), " seconds"))

## 108 images take ~21 seconds locally (5img/s)

##########################################################################################
#### Run the parallel kernel

BATCH_SIZE = 14;                            # 27 is two tasks per node on small dataset and small cluster
                                             # 14 is one tasks per node on small dataset and big cluster
                                             # larger batch size for larger dataset will defray overhead
NO_BATCHES = ceiling(nrow(blob_info)/BATCH_SIZE);


#### cluster execution
start_time <- Sys.time()
results <- foreach(i=1:NO_BATCHES ) %dopar% {     # %dopar% invokes parallel backend (registered cluster)
  N = nrow(blob_info);
  fromRow = (i-1)*BATCH_SIZE+1;
  toRow = min(i*BATCH_SIZE, N);
  parallel_kernel(blob_info[fromRow:toRow,])
}
end_time <- Sys.time()
print(paste0("Ran for ", as.numeric(end_time - start_time, units="secs"), " seconds"))

## 108 images take ~54 seconds on small cluster (2 img/s) 
## 108 images take ~24 seconds on large cluster (5 img/s) (cluster overhead)
##  5k images take ~255 seconds on large cluster (20 img/s)
## 10k images take ~417 seconds on large cluster (23 img/s) 
## 8 nodes -> 4.5x speedup vs local. My desktop is a bigger machine, so great!


##########################################################################################
# clean up result: it's a list of outputs, one from each task, we need to rbind them
single_df <- Reduce(rbind, results)

# only URLs were processed, turn them into person names
single_df$fname <- sapply(strsplit(single_df$url, '/'), function(l) {l[6]})
single_df$bname <- sapply(strsplit(single_df$fname, ".", fixed=TRUE), function(l) l[1])
single_df$pname <- sapply(strsplit(single_df$fname, "_", fixed=TRUE), 
                          function(l) paste(l[1:(length(l)-1)], collapse=" "))

# save results as an Rds
saveRDS(single_df, "faces_featurized_resnet18.Rds")

#####################################################################################
# makes sense?

library(tidyr)
library(ggplot2)
library(magrittr)
features <- single_df %>% gather(featname, featval, -bname, -pname)        # plot features by file
plottable <- features[startsWith(features$featname, 'Feature'),];
plottable$featval <- type.convert(plottable$featval);                       # make numeric again

(
  p <- ggplot(plottable, aes(featname, pname)) + 
    geom_tile(aes(fill = featval), colour = "white") +
    scale_fill_gradient(low = "white",high = "steelblue")
)



