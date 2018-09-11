# Add RServer libraries to the R session config, but only after all libs are installed
# sudo echo "r-libs-user=/data/mlserver/9.2.1/libraries/RServer" >>/etc/rstudio/rsession.conf

# Enable the RStudio Server service
sudo systemctl enable rstudio-server

# Start the RStudio Server service
sudo systemctl start rstudio-server

sudo mkdir /data/active-learning-data

# take ownership of /data/active-learning-data
sudo chown -R remoteuser:remoteuser /data/active-learning-data

# create conda environment
cd text_featurization/lm_finetune
/anaconda/bin/conda env create -f conda.yml
source /anaconda/bin/activate py35
pip install ipykernel
sudo /anaconda/envs/embeddings/bin/python -m ipykernel install --name embeddings --display-name "embeddings"
