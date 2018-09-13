##########################################################################################
# Chapter 1: Using microsoftml to featurize images

"""
The steps:
1) List the contents of the blob using Azure Python SDK into a dataframe
2) Run the featurize function locally
3) Run the featurize function on Azure Batch
"""


####################################################################
# identify where the data is

storageAccount = "storage4tomasbatch";
from secrets import storageKey;
container = "tutorial";
blob_url_base = "https://{}.blob.core.windows.net/{}/".format(storageAccount, container);


####################################################################
# list the blob store
import pandas
import azure.storage.blob

def listFilesInAzureFolder(account_name, account_key, container_name, prefix):

    blob_service = azure.storage.blob.BlockBlobService(account_name=account_name, 
                                account_key=account_key, protocol="https")
            
    blobs = blob_service.list_blobs(container_name, prefix = prefix)    

    # now strip the prefix (directory) and the presumed '/' after it
    blobnames = [blob.name for blob in blobs]    
    filenames = [blob.name[(len(prefix) + 1):] for blob in blobs]
    bloburls = ["https://{}.blob.core.windows.net/{}/{}".format(storageAccount, container, blob.name) for blob in blobs]

    return pandas.DataFrame({"file_name" : filenames, "blob_name": blobnames, "url": bloburls})

data = listFilesInAzureFolder(storageAccount, storageKey, container, 'faces_small');

################################################################
# Parallel kernel

import os
import urllib.request
from microsoftml import load_image, resize_image, extract_pixels, rx_featurize, featurize_image

def parallel_kernel(df, data_dir = None, download_fresh=False):
    """
    Args: 
    df  a data frame with urls and file names, as created by listFilesInAzureFolder
    """

    # get the images from blob and do them locally
    if data_dir is None:
        data_dir <- file.path(os.getcwd(), 'localdata');
    if (not os.path.exists(data_dir)):
       os.makedirs(data_dir);

    # download the assigned blob files, serially to start with
    df["local_name"] = df["file_name"].apply(lambda f: os.path.join(data_dir, f));
    for i in range(len(df)):        
        if (not os.path.exists(df.loc[i, "local_name"])) or download_fresh:
            urllib.request.urlretrieve(df.loc[i, "url"], df.loc[i, "local_name"]);

    # featurize
    image_features = rx_featurize(
        data = df,
        # declare the featurization pipeline
        ml_transforms = [load_image(cols=dict(Image = "local_name")),   # will make a column named "Image"
                        resize_image(cols=dict(Resized = "Image"),      # will make "Resized" from "Image"
                                     width = 224, height = 224,
                                     resizing_option = "IsoPad"),
                        extract_pixels(cols=dict(Pixels = "Resized")), 
                        featurize_image(cols= dict(Features = "Pixels"), dnn_model = "Resnet18")
                       ],
        ml_transform_vars = ["local_name"],       # transform these columns
        report_progress=1)

    image_features.url = df.url;
    return(image_features)

allfeat = parallel_kernel(data, data_dir = "E:\\temp");

###################################################################
## Show it
import matplotlib.pyplot as plt

featcolumns = [c for c in feat.columns if str.startswith(c,'Features')]
feat = allfeat[featcolumns]
plt.imshow(feat, cmap='hot')
plt.show()
