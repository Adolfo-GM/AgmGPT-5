param([switch]$Smoke,[int]$Port=8785,[switch]$Resume,
      [ValidateSet('sft','pretrain')][string]$Stage='sft',
      [string]$DataDir='', [string]$InitFrom='', [int]$Steps=1000,
      [string]$Python='')
$ErrorActionPreference='Stop'
if (-not $Python) {
    $localPython=Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $localPython) { $Python=$localPython }
    else {
        $command=Get-Command python -ErrorAction SilentlyContinue
        if ($command) { $Python=$command.Source }
    }
}
if (-not $Python) { throw 'Activate a Python environment with requirements.txt installed, or supply -Python.' }
$env:PYTHONUTF8='1'
$config=if($Smoke){'config-smoke.json'}else{'config.json'}
Push-Location $PSScriptRoot
try {
    $trainArgs=@('-B','-u','train.py','--config',$config,'--port',"$Port",'--stage',$Stage)
    if($Resume){$trainArgs+='--resume'}
    if($DataDir){$trainArgs+=@('--data-dir',$DataDir)}
    if($InitFrom){$trainArgs+=@('--init-from',$InitFrom)}
    if($Smoke){$trainArgs+=@('--steps','10','--seconds','120','--accumulation','1','--warmup','2')}
    else {$trainArgs+=@('--steps',"$Steps")}
    & $python @trainArgs
    if($LASTEXITCODE -ne 0){throw "Trainer exited with code $LASTEXITCODE"}
} finally { Pop-Location }
