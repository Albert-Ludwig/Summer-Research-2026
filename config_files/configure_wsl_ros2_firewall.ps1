$ErrorActionPreference = "Stop"

$wslCreatorId = "{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}"
$remoteAddress = "192.168.131.1"
$ports = @("7400-7600", "11811")

$hyperVRuleName = "WSL-ROS2-DDS-Jackal-UDP"
$hyperVRule = Get-NetFirewallHyperVRule -Name $hyperVRuleName -ErrorAction SilentlyContinue
if (-not $hyperVRule) {
    New-NetFirewallHyperVRule `
        -Name $hyperVRuleName `
        -DisplayName "WSL ROS 2 DDS from Jackal" `
        -Direction Inbound `
        -VMCreatorId $wslCreatorId `
        -Protocol UDP `
        -LocalPorts $ports `
        -RemoteAddresses $remoteAddress `
        -Action Allow | Out-Null
}

Write-Host "Configured a scoped WSL ROS 2 DDS firewall rule for $remoteAddress."
