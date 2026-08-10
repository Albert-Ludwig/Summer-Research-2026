$ErrorActionPreference = "Stop"

$ruleName = "WSL-RoboHub-noVNC-Internal"
$wslCreatorId = "{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}"
$wslDistribution = "Ubuntu-22.04"
$wslAddresses = (wsl -d $wslDistribution -- hostname -I) -split '\s+'
$wslAddress = $wslAddresses |
    Where-Object { $_ -match '^\d{1,3}(\.\d{1,3}){3}$' -and $_ -notlike '172.17.*' } |
    Select-Object -First 1

if (-not $wslAddress) {
    throw "Could not determine the IPv4 address of WSL distribution '$wslDistribution'."
}

Get-NetFirewallHyperVRule -Name $ruleName -ErrorAction SilentlyContinue |
    Remove-NetFirewallHyperVRule -ErrorAction SilentlyContinue

New-NetFirewallHyperVRule `
    -Name $ruleName `
    -DisplayName "WSL RoboHub noVNC internal" `
    -Direction Inbound `
    -VMCreatorId $wslCreatorId `
    -Protocol TCP `
    -LocalAddresses $wslAddress `
    -RemoteAddresses $wslAddress `
    -LocalPorts 6083 `
    -Action Allow | Out-Null

$windowsRuleName = "WSL-RoboHub-noVNC-Internal-Windows"
Get-NetFirewallRule -Name $windowsRuleName -ErrorAction SilentlyContinue |
    Remove-NetFirewallRule -ErrorAction SilentlyContinue

New-NetFirewallRule `
    -Name $windowsRuleName `
    -DisplayName "WSL RoboHub noVNC internal Windows" `
    -Direction Inbound `
    -Action Allow `
    -Protocol TCP `
    -LocalAddress $wslAddress `
    -RemoteAddress $wslAddress `
    -LocalPort 6083 `
    -Profile Any | Out-Null

Write-Host "Configured noVNC access on http://${wslAddress}:6083/."
