param([Parameter(Mandatory=$true)][string]$OutputDirectory)
$ErrorActionPreference = 'Stop'
$referenceCommit = '4d09ba6d6a7e891dd8063974dfe81b3d78e3494f'
$referenceRoot = 'https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/' + $referenceCommit + '/'
$destinationRoot = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $destinationRoot) { throw 'Output already exists; preserve it and choose a new directory.' }
[IO.Directory]::CreateDirectory($destinationRoot) | Out-Null
$referenceFiles = @(
    'LICENSE',
    'exp5_write_from_scratch/README.md',
    'exp5_write_from_scratch/LICENSE-MinAtar',
    'exp5_write_from_scratch/networks/breakout/hand.npz',
    'exp5_write_from_scratch/eval/minatar_eval.json',
    'exp5_write_from_scratch/runs/breakout/message.txt',
    'exp5_write_from_scratch/runs/breakout/reply/answer.md',
    'exp5_write_from_scratch/runs/breakout/reply/run.json',
    'exp5_write_from_scratch/runs/breakout/apply.json'
)
$tree = Invoke-RestMethod -Uri ('https://api.github.com/repos/RikiyaT/lm_understand-logs/git/trees/' + $referenceCommit + '?recursive=1')
if ($tree.truncated) { throw 'Reference tree was truncated.' }
$records = @()
foreach ($relativePath in $referenceFiles) {
    $entry = @($tree.tree | Where-Object { $_.path -eq $relativePath })
    if ($entry.Count -ne 1 -or $entry[0].type -ne 'blob' -or $entry[0].size -gt 2000000) { throw ('Invalid reference entry: ' + $relativePath) }
    $targetPath = Join-Path $destinationRoot $relativePath
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($targetPath)) | Out-Null
    Invoke-WebRequest -Uri ($referenceRoot + $relativePath) -OutFile $targetPath
    $bytes = [IO.File]::ReadAllBytes($targetPath)
    $header = [Text.Encoding]::UTF8.GetBytes(('blob ' + $bytes.Length + [char]0))
    $sha1 = [Security.Cryptography.SHA1]::Create()
    try { $blobHash = ([BitConverter]::ToString($sha1.ComputeHash($header + $bytes))).Replace('-','').ToLowerInvariant() }
    finally { $sha1.Dispose() }
    if ($blobHash -ne $entry[0].sha) { throw ('Git blob identity differs: ' + $relativePath) }
    $records += [ordered]@{ path=$relativePath; bytes=$bytes.Length; git_blob_sha1=$blobHash; sha256=(Get-FileHash -LiteralPath $targetPath -Algorithm SHA256).Hash.ToLowerInvariant() }
}
$receipt = [ordered]@{
    format='wspl.csail_breakout_preparation.v0'; status='complete'; reference_commit=$referenceCommit
    source_repository='https://github.com/RikiyaT/lm_understand-logs'; files=$records
    preparation_source_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    scope='Download and identity checks only; no policy evaluation, model calls or training.'
    license_note='Upstream logs/networks CC BY 4.0; scripts MIT; MinAtar source in prompt GPL-3.0. Preserve downloaded notices.'
}
$json = ($receipt | ConvertTo-Json -Depth 7) + "`n"
[IO.File]::WriteAllText((Join-Path $destinationRoot 'bundle.json'), $json, (New-Object Text.UTF8Encoding($false)))
Write-Output ('Prepared reference bundle: ' + $destinationRoot)
