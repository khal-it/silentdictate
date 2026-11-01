# Report whether the VB-CABLE devices are present. Exits 0 if found, 1 if not.
$devices = Get-CimInstance Win32_SoundDevice | Where-Object { $_.Name -like '*VB-Audio*' -or $_.Name -like '*CABLE*' }

if ($devices) {
    $devices | ForEach-Object { Write-Output "Found: $($_.Name) [$($_.Status)]" }
    Write-Output 'SilentDictate outputs to "CABLE Input"; other apps select "CABLE Output".'
    exit 0
}

Write-Output 'VB-CABLE was not found. See scripts\setup_virtual_mic_windows.md.'
exit 1
