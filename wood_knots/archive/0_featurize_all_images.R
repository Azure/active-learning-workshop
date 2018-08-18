# sCRIPT that featurizes all images

# DATA SOURCE:
# 
# These images are from the University of Oulu, Finland (http://www.ee.oulu.fi/~olli/Projects/Lumber.Grading.html).
# The labelled images (http://www.ee.oulu.fi/research/imag/knots/KNOTS) were saved as individual knot images by the original authors, 
# and we segmented the "unlabelled" images by hand using LabelImg (https://github.com/tzutalin/labelImg). 
# We have converted all of the individual knot images to PNG format, and in this script we download zip files containing PNG versions
# of the labelled images and the segmented unlabelled images from Azure blob storage.

.libPaths( c( "/data/mlserver/9.2.1/libraries/RServer", .libPaths()))
library(RevoScaleR)
library(MicrosoftML)

DATA_DIR <- file.path(getwd(), 'data')

LABELLED_FEATURIZED_DATA <- file.path(DATA_DIR, "labelled_knots_featurized_resnet18.Rds")
UNLABELLED_FEATURIZED_DATA <- file.path(DATA_DIR, "unlabelled_knots_featurized_resnet18.Rds")

LABELLED_IMAGE_DIR <- file.path(DATA_DIR, "knot_images_png")
UNLABELLED_IMAGE_DIR <- file.path(DATA_DIR, "unlabelled_cropped_png")

LABELS_FILE <- file.path(DATA_DIR, "names.txt")

# All knot classes
KNOT_CLASSES <- setNames(nm=c("sound_knot", "dry_knot", "encased_knot"))


if(!dir.exists(DATA_DIR)) dir.create(DATA_DIR)

labelled_image_url <- 'https://isvdemostorageaccount.blob.core.windows.net/wood-knots/labelled_knot_images_png.zip'
unlabelled_image_url <- 'https://isvdemostorageaccount.blob.core.windows.net/wood-knots/unlabelled_cropped_png.zip'
names_url <- 'http://www.ee.oulu.fi/research/imag/knots/KNOTS/names.txt'

# Download and unzip image files
download.file(labelled_image_url, destfile = file.path(DATA_DIR, 'knot_images_png.zip'))
download.file(unlabelled_image_url, destfile = file.path(DATA_DIR, 'unlabelled_cropped_png.zip'))
download.file(names_url, destfile = file.path(DATA_DIR, 'names.txt'))
unzip(file.path(DATA_DIR, 'knot_images_png.zip'), exdir = DATA_DIR)
unzip(file.path(DATA_DIR, 'unlabelled_cropped_png.zip'), exdir = DATA_DIR)


# I have a beta version that capitalized the model name. This should (?) work for other folks.
DNN_MODEL <- if ("Microsoft R Server version 9.2.0.2731 (2017-07-26 06:17:26 UTC)" == Revo.version$version.string){
  "Resnet18"
} else {
  "resnet18"
}


featurize_directory <- function(dir){
  pwd <- setwd(dir)
  knot_info <- data.frame(path = list.files(pattern="*.png"), stringsAsFactors=FALSE)
  
  image_features <- rxFeaturize(data = knot_info,
                                mlTransforms = list(loadImage(vars = list(Image = "path")),
                                                    resizeImage(vars = list(Features = "Image"), 
                                                                width = 224, height = 224, 
                                                                resizingOption = "IsoPad"),
                                                    extractPixels(vars = "Features"),
                                                    featurizeImage(var = "Features", 
                                                                   dnnModel = DNN_MODEL)),
                                mlTransformVars = c("path"))
  
  setwd(pwd)
  image_features
}

# Note: be sure the file name has not been converted to a factor! If it has, you get an error like this:
# Exception: 'Source column 'path' has invalid type ('Key<U4, 0-595>'): Expected Text type.

if( file.exists(UNLABELLED_FEATURIZED_DATA)){
  unlabelled_knot_data_df <- readRDS(UNLABELLED_FEATURIZED_DATA)
} else {
  unlabelled_knot_data_df <- featurize_directory(UNLABELLED_IMAGE_DIR)
  saveRDS(unlabelled_knot_data_df, UNLABELLED_FEATURIZED_DATA)
}

if( file.exists(LABELLED_FEATURIZED_DATA)){
  labelled_knot_data_df <- readRDS(LABELLED_FEATURIZED_DATA)
} else {
  labelled_knot_data_df <- featurize_directory(LABELLED_IMAGE_DIR)
  
  # Add labels to labelled dataset
  labels <- read.table(LABELS_FILE, header=FALSE, sep=" ", stringsAsFactors=FALSE)[1:2]
  names(labels) <- c("path", "knot_class")
  labels$path <- gsub("ppm$", "png", labels$path)
  rownames(labels) <- labels$path
  
  labelled_knot_data_df$knot_class <- labels[labelled_knot_data_df$path, "knot_class"]
  labelled_knot_data_df <- labelled_knot_data_df[labelled_knot_data_df$knot_class %in% KNOT_CLASSES,]
  labelled_knot_data_df$knot_class <- factor(as.character(labelled_knot_data_df$knot_class), levels=KNOT_CLASSES)
  
  saveRDS(labelled_knot_data_df, LABELLED_FEATURIZED_DATA)
}
