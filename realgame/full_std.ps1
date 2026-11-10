# One full STANDARD-mode game (lives + continues) at a given difficulty in AI mode (modding/gamemode.py ai, perception v5):
# play from stage 1 until the gameplay ends (ending / final game over), accepting every continue prompt; count deaths.
# The game is left OPEN at the end (so the replay-save screens can be handled), unless -close.
# Usually started by ..\run.ps1. Output: <TOUHOU_RUNS>\<name>\ (summary.txt, run\, end_screen.png). Python: TOUHOU_PY.
#   .\full_std.ps1 -name test1 -diff 3 -continues 0 -close
param([string]$ckpt = (Join-Path $PSScriptRoot "..\models\v11a_policy.pt"), [string]$name = "std_test", [int]$diff = 3,
      [int]$secs = 3600, [int]$continues = 30, [string]$extra = "--align --items", [switch]$close, [string]$mode = "ai")
$py   = if ($env:TOUHOU_PY) { $env:TOUHOU_PY } else { "python" }
$runs = if ($env:TOUHOU_RUNS) { $env:TOUHOU_RUNS } else { "C:\touhou-runs" }
$out  = Join-Path $runs $name
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
New-Item -ItemType Directory -Force $out | Out-Null
$sum = Join-Path $out "summary.txt"
"" | Set-Content $sum
function Log($m) { "[$(Get-Date -Format s)] $m" | Add-Content $sum }
& $py -c "import time, winutil as W; t0=time.time()
while W.user_idle_seconds() < 20 and time.time()-t0 < 600: time.sleep(2)
print('idle', round(W.user_idle_seconds()))" | ForEach-Object { Log $_ }
& $py closegame.py | Out-Null
& $py ..\modding\gamemode.py $mode 2>&1 | ForEach-Object { Log ("mode: " + $_) }     # (-mode display: the original look, for checks)
$ex = @(); if ($extra) { $ex = $extra -split ' ' }
Log "##### run"
# play.py loads and warms up the models first, then starts the game itself (newgame.py, 2 tries) and takes over at once:
# starting the game before play.py had loaded left the player idle for ~10 s and LUNATIC killed it (2026-10-27)
& $py play.py $ckpt --newgame "$diff standard $out" --seconds $secs --out (Join-Path $out "run") --snap_every 30 --shoot --percep 5 --standard --continues $continues `
    --end_after_no_field 40 @ex 2>&1 |
    Select-String "newgame:|CONTINUE|continue prompt|GAME OVER|STANDARD|GAMEPLAY ENDED|ran |ms per step|dialogue segments|policy:|boss-alignment|enemy-alignment|item-seeking|item type|read_items|bomb|BOMB|register|reflex|perception from memory|cpu boost|Traceback|Error" |
    ForEach-Object { $_.ToString() } | Add-Content $sum
& $py ui.py (Join-Path $out "end_screen.png") | Out-Null
if ($close) { & $py closegame.py | Out-Null; & $py ..\modding\gamemode.py display | Out-Null }
Log "done"
