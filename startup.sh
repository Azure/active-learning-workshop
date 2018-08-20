# Add RServer libraries to the R session config, but only after all libs are installed
# sudo echo "r-libs-user=/data/mlserver/9.2.1/libraries/RServer" >>/etc/rstudio/rsession.conf

# Enable the RStudio Server service
sudo systemctl enable rstudio-server

# Start the RStudio Server service
sudo systemctl start rstudio-server

# take ownership of /data
sudo chown -R remoteuser:remoteuser /data

# create conda environment
cd text_featurization/lm_finetune
conda env create -f conda.yml
source activate py35
pip install ipykernel
sudo /anaconda/envs/embeddings/bin/python -m ipykernel install --name embeddings --display-name "tensorflow"
