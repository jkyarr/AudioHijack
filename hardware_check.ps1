Write-Output "
=== AudioHijack Windows Hardware Assessment ===
"

# Check RAM
$ram = (Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum).Sum / 1GB
Write-Output "RAM: $($ram.ToString('F1')) GB"
if ($ram -ge 32) { Write-Output "✅ RAM sufficient" } else { Write-Output "❌ RAM below 32GB recommended" }

# Check Disk space (C drive)
$disk = Get-PSDrive C
$freeGB = [math]::Round($disk.Free / 1GB, 1)
Write-Output "Free space on C: $($freeGB) GB"
if ($freeGB -ge 100) { Write-Output "✅ Disk space OK" } else { Write-Output "⚠️  Low disk space - need ~1TB recommended" }

# GPU and CUDA check
try {
    $gpu = Get-CimInstance Win32_VideoController | Where-Object {$_.Name -like "*NVIDIA*"}
    if ($gpu) {
        Write-Output "GPU: $($gpu.Name)"
        Write-Output "VRAM: ~$([math]::Round($gpu.AdapterRAM / 1GB)) GB (approximate)"
    } else {
        Write-Output "No NVIDIA GPU detected"
    }
} catch {
    Write-Output "Could not detect GPU details"
}

Write-Output "
Run 'nvidia-smi' in a new terminal for full CUDA/VRAM details if you have NVIDIA GPU."
Write-Output "
Paste this entire output back to Grok for assessment."
