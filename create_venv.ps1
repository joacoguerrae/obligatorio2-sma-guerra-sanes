param(
    [string]$VenvDir = ".venv"
)

Write-Host "Creando virtual environment en: $VenvDir"
python -m venv $VenvDir
& "$VenvDir\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
& "$VenvDir\Scripts\python.exe" -m pip install -e .
Write-Host "Entorno creado. Activa con: $VenvDir\\Scripts\\Activate.ps1"
