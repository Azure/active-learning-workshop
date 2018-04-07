param([string]$clustername, [string]$resourcegroup, [string]$profilepath, [string]$clusterpasswd, [string]$sshusername)

function SubmitDSVMCreation {
    param([string]$clustername, [string]$resourcegroup, [string]$clusterpasswd, [string]$sshusername)
	$templatePath = "D:\git\kdd\template.json"; 
    Select-AzureRmProfile -Path $profilepath;
	$dsvmparams = @{rg=$resourcegroup;storageAccountType="Premium_LRS";location="westus";virtualMachineSize="Standard_DS12_v2";virtualMachineName=$clustername;adminUsername=$sshusername;virtualNetworkName=$clustername+"-vnet";networkInterfaceName=$clustername+"359";networkSecurityGroupName=$clustername+"-ng";adminPassword=$clusterpasswd;storageAccountName=$clustername+"disk";publicIpAddressName=$clustername+"-ip"};
	New-AzureRmResourceGroupDeployment -Name $clustername -ResourceGroupName $resourcegroup -TemplateFile $templatePath -TemplateParameterObject $dsvmparams;
}
SubmitDSVMCreation -clustername $clustername -resourcegroup $resourcegroup -clusterpasswd $clusterpasswd -sshusername $sshusername
