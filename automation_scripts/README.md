# Automation Scripts

The automation scripts in this directory can be used to provision many Data Science VMs in parallel. These can be used if you are teaching a hands-on tutorial and wish to provide each student with a VM.

## Instructions

1. Install Azure PowerShell - see https://docs.microsoft.com/en-us/powershell/azure/install-azurerm-ps 
1.	Clone or download https://github.com/Azure/active-learning-workshop
1.	Configure the provisioning by editing automation_scripts/ClusterParameters.csv, providing values for SubscriptionID, ResourceGroup, ClusterStartIndex, ClusterEndIndex, and ClusterPrefix
1. Create a Resource Group with the name you entered in ClusterParameters.csv
1.	In Azure PowerShell, run the provisioning script

```bash
cd active-learning-workshop\automation_scripts
.\RunDSVMCreation.ps1
```