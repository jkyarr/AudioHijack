# AudioHijack Cloud GPU Setup for Windows
# Created via Grok GitHub Connector for low-end hardware

Write-Host '=== AudioHijack Cloud GPU Setup ===' -ForegroundColor Cyan

Write-Host 'Your hardware:' -ForegroundColor Yellow
Write-Host 'RAM: 16GB (insufficient)' 
Write-Host 'GPU: RTX 4060 Laptop (~4-8GB VRAM) - insufficient for full attack' 

Write-Host ''
Write-Host 'Recommended: Use a cloud GPU with 48GB+ VRAM (A100 / H100 / RTX 6000 Ada)' -ForegroundColor Green

Write-Host ''
Write-Host 'Cheapest options (as of May 2026):' -ForegroundColor White
Write-Host '1. RunPod.io - A100 80GB ~ $0.79/hr' 
Write-Host '2. Vast.ai - Often cheaper spot instances' 
Write-Host '3. Lambda Labs or Massed Compute' 

Write-Host ''
Write-Host 'Quick start for RunPod:' -ForegroundColor Cyan
Write-Host '1. Go to https://runpod.io' 
Write-Host '2. Deploy a Pod with PyTorch + CUDA 12.4 template' 
Write-Host '3. Choose A100 40GB or better (80GB preferred)' 
Write-Host '4. Once running, run the following in the pod terminal:' 
Write-Host '   git clone https://github.com/jkyarr/AudioHijack.git' 
Write-Host '   cd AudioHijack' 
Write-Host '   ./windows_setup.ps1   # or the linux equivalent' 

Write-Host ''
Write-Host 'After cloud setup, use this command to start attack:' -ForegroundColor Green
Write-Host 'python run_attack.py lalm=voxtral_mini attack=caa' 

Write-Host ''
Write-Host 'Would you like me to also create a minimal_local_test.ps1 for your laptop?' -ForegroundColor White