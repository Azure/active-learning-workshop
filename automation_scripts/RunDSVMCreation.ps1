#######################################################################
# FIRST LOGIN AND SAVE PROFILE IN A FILE WHICH WILL BE USED FOR SUBMITTING JOBS IN PARALLEL
#######################################################################
Login-AzureRmAccount

#######################################################################
# SET WORKING FOLDER PATH
#######################################################################
#Set-Location -Path C:\Users\deguhath\Desktop\CDSP\Spark\KDDBlog\ProvisionScripts
$basepath = Get-Location
#######################################################################
# READ IN CLUSTER CONFIGURATION PARAMETERS
#######################################################################
#READ IN PARAMETER CSV CONFIGURATION FILE
$currentPathTmp = Get-Location
$currentPath = [string]$basepath
$paramFile = Import-Csv ClusterParameters.csv
echo $paramFile
# GET ALL THE PARAMETERS FROM THE CONFIGURATION PARAMETER FILE
$scriptpathnametmp = $paramFile | Where-Object {$_.Parameter -eq "ScriptPath"} | % {$_.Value}
$scriptpathname = [string]$scriptpathnametmp
$subscriptionname = $paramFile | Where-Object {$_.Parameter -eq "SubscriptionName"} | % {$_.Value}
$subscriptionid = $paramFile | Where-Object {$_.Parameter -eq "SubscriptionID"} | % {$_.Value}
$tenantid = $paramFile | Where-Object {$_.Parameter -eq "TenantID"} | % {$_.Value}
$resourcegroup = $paramFile | Where-Object {$_.Parameter -eq "ResourceGroup"} | % {$_.Value}
$profilepathtmp = $paramFile | Where-Object {$_.Parameter -eq "Profilepath"} | % {$_.Value}
$profilepath = $currentPath + "\" + [string]$profilePathTmp
$tmp = $paramFile | Where-Object {$_.Parameter -eq "ClusterStartIndex"} | % {$_.Value}; $clusterstartindex = [int]$tmp;
$tmp = $paramFile | Where-Object {$_.Parameter -eq "ClusterEndIndex"} | % {$_.Value}; $clusterendindex = [int]$tmp;
$clusterprefix = $paramFile | Where-Object {$_.Parameter -eq "ClusterPrefix"} | % {$_.Value}
$tmp = $paramFile | Where-Object {$_.Parameter -eq "ClusterInfoOutputFileName"} | % {$_.Value}; $clusterinfooutputfilename = $currentPath + "\" + [string]$tmp 
$tmp = $paramFile | Where-Object {$_.Parameter -eq "NumWorkerNodes"} | % {$_.Value}; $numworkernodes = [int]$tmp;
$tmp = $paramFile | Where-Object {$_.Parameter -eq "AdminUsername"} | % {$_.Value}; $adminusername = [string]$tmp;
$tmp = $paramFile | Where-Object {$_.Parameter -eq "SshUsername"} | % {$_.Value}; $sshusername = [string]$tmp;

#######################################################################
# SET AZURE SUBSCRIPTIONS & SAVE PROFILE INFORMATION IN A FILE
#######################################################################
echo $subscriptionid
Set-AzureRmContext -SubscriptionID $subscriptionid #-TenantID $tenantid
Save-AzureRmProfile -Path $profilepath

#######################################################################
# SUBMIT CLUSTER JOBS TO BE RUN IN PARALLEL
# NOTE: YOU NEED TO SPECIFY THE LOCATION OF FUNCTION FILE, C:\Users\deguhath\Source\Repos\KDD2016 Spark\Scripts\SubmitHDICreation.ps1 
#######################################################################
for($i=$clusterstartindex; $i -le $clusterendindex; $i++) {
	$clustername = $clusterprefix+$i;
	# Generates a random number between 1 and 10000 and adds it to the clustername for the cluster password 
	$randnum = Get-Random -minimum 1 -maximum 100001
	$clusterpasswd = $clustername.substring(0,1).toupper() + $clustername.substring(1).tolower() + "_" + $randnum
	$dns = "ssh $sshusername@$clustername.westus.cloudapp.azure.com"
	# Outputs the cluster login and password information to a file
	Set-Location -Path $currentPath
	$clustername,$clusterpasswd,$dns -join ',' | out-file -filepath $clusterinfooutputfilename -append -Width 200;
	
	## SPECIFY ABSOLUTE OR RELATIVE PATH TO THE SCRIPT THAT YOU WANT TO SUBMIT AND RUN IN PARALLEL
	Start-Job -FilePath SubmitDSVMCreation.ps1 -ArgumentList @($clustername, $resourcegroup, $profilepath, $clusterpasswd, $sshusername, $currentPath)
}
#######################################################################
# END
#######################################################################
