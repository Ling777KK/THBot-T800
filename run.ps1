<#
.SYNOPSIS
  Let the bot play Touhou EoSD New Classic by itself (Standard mode, no continues). One command; see README.md.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\run.ps1
      one LUNATIC game with the setup of the clears (memory perception + reflex layer, bombs on)
  powershell -ExecutionPolicy Bypass -File .\run.ps1 -Games 5
      up to 5 games; stops at the first clear and leaves the game open so the replay can be saved
  powershell -ExecutionPolicy Bypass -File .\run.ps1 -Difficulty NORMAL -NoBomb
  powershell -ExecutionPolicy Bypass -File .\run.ps1 -Vision
      the pure-vision variant (bullets from the picture only)
  powershell -ExecutionPolicy Bypass -File .\run.ps1 -Check
      only check Python, the packages, the GPU and the game folder
  powershell -ExecutionPolicy Bypass -File .\run.ps1 -Restore
      close the game and switch the textures back to the original look (after Ctrl+C or after saving a replay)
#>
param(
    [ValidateSet("EASY", "NORMAL", "HARD", "LUNATIC")][string]$Difficulty = "LUNATIC",
    [int]$Games = 1,
    [switch]$Vision,
    [switch]$NoBomb,
    [switch]$Check,
    [switch]$Restore,
    [string]$Out = $(if ($env:TOUHOU_RUNS) { $env:TOUHOU_RUNS } else { "C:\touhou-runs" }),
    [string]$GameDir = $env:TH06NC_DIR,
    [string]$Python = $env:TOUHOU_PY
)

$root = $PSScriptRoot
$env:TOUHOU_RUNS = $Out
if ($GameDir) { $env:TH06NC_DIR = $GameDir }

function Say($m, $color = "Gray") { Write-Host $m -ForegroundColor $color }

# --- 1. a Python that has PyTorch with CUDA and the other packages ------------------------------------------------------
$probe = "import sys, torch, mss, numpy, PIL, zstandard; print(sys.executable + '|' + str(torch.cuda.is_available()))"
$cands = @()
if ($Python) { $cands += , @($Python) }
$cands += , @("py", "-3.12")
$cands += , @("py", "-3")
$cands += , @("python")
Get-ChildItem "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe", "C:\Program Files\Python3*\python.exe" -ErrorAction SilentlyContinue |
    Sort-Object FullName -Descending | ForEach-Object { $cands += , @($_.FullName) }
$py = $null; $noCuda = $null
foreach ($c in $cands) {
    $exe = $c[0]; $pre = @($c | Select-Object -Skip 1)
    try { $r = & $exe @pre -c $probe 2>$null } catch { continue }
    if ($LASTEXITCODE -ne 0 -or -not $r) { continue }
    $parts = "$r".Trim().Split("|")
    if ($parts[1] -eq "True") { $py = $parts[0]; break }
    if (-not $noCuda) { $noCuda = $parts[0] }
}
if (-not $py) {
    if ($noCuda) { Say "PyTorch in $noCuda cannot see a CUDA GPU: install the CUDA build of PyTorch (pytorch.org); an NVIDIA GPU is needed." Red }
    else { Say "No Python with the needed packages found. Install Python 3.12, then: pip install -r requirements.txt (and PyTorch with CUDA)." Red
           Say "Or pass the interpreter: -Python C:\path\to\python.exe" Red }
    exit 1
}
$env:TOUHOU_PY = $py
Say "Python:      $py" Green

# --- 2. the game folder (gamemode.py finds it in the Steam libraries, or TH06NC_DIR / -GameDir) ---------------------------
$st = & $py "$root\modding\gamemode.py" status 2>&1
if ($LASTEXITCODE -ne 0) { Say "$st" Red; Say "Pass the folder that holds th06nc.exe with -GameDir." Red; exit 1 }
Say "Game:        found (textures: $(($st | Select-Object -First 1)))" Green
Say "Output:      $Out" Green

if ($Restore) {
    Push-Location "$root\realgame"
    & $py closegame.py
    & $py ..\modding\gamemode.py display
    Pop-Location
    exit 0
}
if ($Check) { Say "Everything needed is there." Green; exit 0 }

# --- 3. play --------------------------------------------------------------------------------------------------------------
$diff = @{ EASY = 0; NORMAL = 1; HARD = 2; LUNATIC = 3 }[$Difficulty]
# (paths relative to realgame\, where full_std.ps1 runs: the extra options are split on spaces)
if ($Vision) {
    $extra = "--align --items --traj ..\models\rnn0_traj.pt --traj_margin 0.2 --tint_q 0.1 --vision_reflex --reflex 10 --reflex_hold 4"
    if (-not $NoBomb) { $extra += " --reflex_bomb --reflex_bomb_sum 3" }
} else {
    $extra = "--align --enemy_align --traj ..\models\rnn0_traj.pt --traj_margin 0.2 --tint_q 0.1 --memperc ..\models\memperc_fit_types2.json --reflex 10 --reflex_hold 4 --seek_bombs"
    if (-not $NoBomb) { $extra += " --reflex_bomb" }
}
$ckpt = Join-Path $root "models\v11a_policy.pt"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
Say ""
Say "$Games game(s), $Difficulty, standard mode, no continues, $(if ($Vision) { 'pure vision' } else { 'memory perception' }), bombs $(if ($NoBomb) { 'off' } else { 'on' })." Cyan
Say "Hands off the keyboard and mouse: each game starts after 20 s without input, then the game is (re)started through Steam." Cyan
Say "Steam must be running. Ctrl+C stops; afterwards run this script with -Restore." Cyan
for ($i = 1; $i -le $Games; $i++) {
    $name = "run_${stamp}_$i"
    Say ""
    Say "Game $i / $Games -> $Out\$name   (started $(Get-Date -Format 'HH:mm:ss'))" Yellow
    & "$root\realgame\full_std.ps1" -ckpt $ckpt -name $name -diff $diff -continues 0 -extra $extra | Out-Null
    $sumf = Join-Path $Out "$name\summary.txt"
    $s = if (Test-Path $sumf) { Get-Content $sumf -Raw } else { "" }
    ($s -split "`r?`n") | Where-Object { $_ -match "GAME OVER|GAMEPLAY ENDED|BOMBS:|ran |Traceback|Error" } | ForEach-Object { Say "   $_" }
    if ($s -match "GAME OVER") {
        Say "   game over" Yellow
        Push-Location "$root\realgame"; & $py closegame.py | Out-Null; Pop-Location
        continue
    }
    if ($s -match "GAMEPLAY ENDED") {
        Say "   The game ended without a game over: probably a CLEAR. The game is left open." Green
        Say "   Press Z through the ending and the staff roll, then save the replay into an EMPTY slot (do not overwrite yours)." Green
        Say "   Afterwards run:  powershell -ExecutionPolicy Bypass -File .\run.ps1 -Restore" Green
        exit 0
    }
    Say "   The game ended in an unexpected way (see $sumf); stopping." Red
    break
}
Push-Location "$root\realgame"
& $py closegame.py | Out-Null
& $py ..\modding\gamemode.py display | Out-Null
Pop-Location
Say ""
Say "Done: game closed, original textures restored. Results in $Out\run_${stamp}_*\summary.txt" Green
