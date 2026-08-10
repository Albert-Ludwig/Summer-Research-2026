#Requires -RunAsAdministrator

[CmdletBinding()]
param(
    [string]$ExistingAddress = "192.168.2.10",
    [string]$JackalHostAddress = "192.168.131.101",
    [int]$PrefixLength = 24,
    [string]$JackalAddress = "192.168.131.1"
)

$ErrorActionPreference = "Stop"

$current = Get-NetIPAddress `
    -AddressFamily IPv4 `
    -IPAddress $ExistingAddress `
    -ErrorAction SilentlyContinue | Select-Object -First 1

if (-not $current) {
    throw "Could not find the Ethernet interface with address $ExistingAddress."
}

$adapter = Get-NetAdapter -InterfaceIndex $current.InterfaceIndex
if ($adapter.Status -ne "Up") {
    throw "Adapter '$($adapter.Name)' is not up. Check the Ethernet cable."
}

$configured = Get-NetIPAddress `
    -InterfaceIndex $adapter.ifIndex `
    -AddressFamily IPv4 `
    -ErrorAction SilentlyContinue | Where-Object IPAddress -eq $JackalHostAddress

if (-not $configured) {
    New-NetIPAddress `
        -InterfaceIndex $adapter.ifIndex `
        -IPAddress $JackalHostAddress `
        -PrefixLength $PrefixLength `
        -AddressFamily IPv4 | Out-Null
}

Write-Host "Adapter: $($adapter.Name)"
Get-NetIPAddress -InterfaceIndex $adapter.ifIndex -AddressFamily IPv4 |
    Format-Table IPAddress, PrefixLength, AddressState -AutoSize

if (Test-Connection -ComputerName $JackalAddress -Count 2 -Quiet) {
    Write-Host "PASS: Jackal $JackalAddress is reachable." -ForegroundColor Green
} else {
    Write-Warning "The IP was configured, but Jackal $JackalAddress did not answer ping."
    Write-Warning "Check that the Jackal is powered on and the cable is connected to its Ethernet port."
}
