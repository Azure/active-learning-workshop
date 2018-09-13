# Featurized data and labels

These files contain the numerical features computed for each image in the featurization exercise, as well as the class labels.

These images were from the [University of Oulu](http://www.ee.oulu.fi/~olli/Projects/Lumber.Grading.html), Finland. 
The [labelled images](http://www.ee.oulu.fi/research/imag/knots/KNOTS) were saved as individual knot images by the original authors, 
and we segmented the "unlabelled" images by hand using [LabelImg](https://github.com/tzutalin/labelImg), so that we could preserve the
label associated with each knot.

We have converted all of the individual knot images to PNG format, and you can download zip files containing PNG versions of 
the [labelled images](https://isvdemostorageaccount.blob.core.windows.net/wood-knots/labelled_knot_images_png.zip) and 
the [segmented unlabelled images](https://isvdemostorageaccount.blob.core.windows.net/wood-knots/unlabelled_cropped_png.zip) 
from Azure blob storage.
