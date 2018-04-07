# Add RServer libraries to the R session config, but only after all libs are installed
# sudo echo "r-libs-user=/data/mlserver/9.2.1/libraries/RServer" >>/etc/rstudio/rsession.conf

# Enable the RStudio Server service
sudo systemctl enable rstudio-server

# Start the RStudio Server service
sudo systemctl start rstudio-server
