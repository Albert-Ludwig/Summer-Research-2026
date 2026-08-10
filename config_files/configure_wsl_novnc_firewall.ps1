$ErrorActionPreference = "Stop"

$ruleName = "WSL-RoboHub-noVNC-Loopback"
$wslCreatorId = "{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}"
$rule = Get-NetFirewallHyperVRule -Name $ruleName -ErrorAction SilentlyContinue

if (-not $rule) {
    New-NetFirewallHyperVRule `
        -Name $ruleName `
        -DisplayName "WSL RoboHub noVNC loopback" `
        -Direction Inbound `
        -VMCreatorId $wslCreatorId `
        -Protocol TCP `
        -LocalAddresses 127.0.0.1 `
        -LocalPorts 6083 `
        -Action Allow | Out-Null
}

Write-Host "Configured WSL loopback access for noVNC on TCP 6083."
