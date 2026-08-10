$ErrorActionPreference = "Stop"

$listenAddress = "127.0.0.1"
$listenPort = 6083
$wslDistribution = "Ubuntu-22.04"
$connectPort = 6083

$wslAddresses = (wsl -d $wslDistribution -- hostname -I) -split '\s+'
$connectAddress = $wslAddresses |
    Where-Object { $_ -match '^\d{1,3}(\.\d{1,3}){3}$' -and $_ -notlike '172.17.*' } |
    Select-Object -First 1

if (-not $connectAddress) {
    throw "Could not determine the IPv4 address of WSL distribution '$wslDistribution'."
}

netsh interface portproxy delete v4tov4 `
    listenaddress=$listenAddress `
    listenport=$listenPort 2>$null | Out-Null

netsh interface portproxy add v4tov4 `
    listenaddress=$listenAddress `
    listenport=$listenPort `
    connectaddress=$connectAddress `
    connectport=$connectPort | Out-Null

Write-Host "Configured http://127.0.0.1:6083/ -> http://${connectAddress}:6083/."
