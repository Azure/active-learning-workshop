# Featurize Images on Azure Batch

## Visualize Features with TensorBoard Project

We can now visualize the featurized images by projecting onto 2 - 3 dimensions using a dimensionality reduction technique. We'll use tensorboard for interactive visualizations. Within the `featurize_images` directory there is a subdirectory called `log`. This contains the logs of the featurized images and their metadata for visualization on tensorboard. To start tensorboard:


```bash
source activate py35
cd ~/notebooks/Strata2018/featurize_images/
tensorboard --logdir=log
```

Now naviate to `http://hostname:6006`.

Alternatively, you can upload the datasets in `Strata2018/featurize_images/caltech_db` to the [online tensorflow projector](http://projector.tensorflow.org/) and visualize it in your browser locally, or simply [follow this link](http://projector.tensorflow.org/?config=https://raw.githubusercontent.com/Azure/Strata2018/master/featurize_images/caltech_tb/caltech_items_config.json).
