param([Parameter(Mandatory=$true)][string]$OutputDirectory)
$ErrorActionPreference='Stop'
$commit='4d09ba6d6a7e891dd8063974dfe81b3d78e3494f'
$destination=[IO.Path]::GetFullPath($OutputDirectory)
if(Test-Path -LiteralPath $destination){throw 'Output exists; preserve prior acquisition'}
$tree=Invoke-RestMethod -Uri "https://api.github.com/repos/RikiyaT/lm_understand-logs/git/trees/${commit}?recursive=1"
if($tree.truncated){throw 'Truncated reference tree'}
$files=@($tree.tree | Where-Object {
    $_.type -eq 'blob' -and (
    $_.path -eq 'LICENSE' -or
    $_.path -eq 'exp2_diffusion/README.md' -or
    $_.path -match '^exp2_diffusion/(networks|eval)/' -or
    $_.path -match '^exp2_diffusion/runs/[^/]+/(prompt.txt|run.json)$' -or
    $_.path -eq 'exp6_go_from_scratch/README.md' -or
    $_.path -match '^exp6_go_from_scratch/networks/(budget_0|budget_50|ppo_same_size_lr3e-5_s[012])/' -or
    $_.path -match '^exp6_go_from_scratch/runs/budget_(0|50)/(message.txt|tool_calls.jsonl|final_answer.md)$' -or
    $_.path -match '^exp6_go_from_scratch/eval/budget_(0|50)/eval_submission.json$' -or
    $_.path -match '^exp6_go_from_scratch/eval/ppo_same_size/ppo_big_eval/.*\.(py|json)$' -or
    $_.path -eq 'harness/sandbox.py')
})
if(($files | Measure-Object size -Sum).Sum -gt 50000000){throw 'Acquisition exceeds50MB bound'}
[IO.Directory]::CreateDirectory($destination) | Out-Null
$records=@()
foreach($entry in $files){
    if($entry.size -gt 2000000){throw 'Unexpected large reference file'}
    $target=Join-Path $destination $entry.path
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($target)) | Out-Null
    Invoke-WebRequest -Uri "https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/$commit/$($entry.path)" -OutFile $target
    $bytes=[IO.File]::ReadAllBytes($target)
    $header=[Text.Encoding]::UTF8.GetBytes(('blob '+$bytes.Length+[char]0))
    $algorithm=[Security.Cryptography.SHA1]::Create()
    try{$blob=([BitConverter]::ToString($algorithm.ComputeHash($header+$bytes))).Replace('-','').ToLowerInvariant()}finally{$algorithm.Dispose()}
    if($blob -ne $entry.sha -or $bytes.Length -ne $entry.size){throw ('Identity mismatch: '+$entry.path)}
    $records += [ordered]@{path=$entry.path;bytes=$bytes.Length;git_blob_sha1=$blob;sha256=(Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()}
}
$receipt=[ordered]@{status='complete';reference_commit=$commit;source_repository='https://github.com/RikiyaT/lm_understand-logs';files=$records;preparation_source_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant();scope='Acquisition and identity checks only; no downloaded code executed, no sampling, model calls or training.';license_note='Upstream logs/data/networks CC BY4.0; scripts MIT. Preserve LICENSE. Published author code and evaluator answer keys are not proposer inputs.'}
[IO.File]::WriteAllText((Join-Path $destination 'bundle.json'),(($receipt | ConvertTo-Json -Depth 8)+"`n"),[Text.UTF8Encoding]::new($false))
Write-Output ("Verified files: "+$records.Count+"; bytes: "+(($records | ForEach-Object { $_['bytes'] } | Measure-Object -Sum).Sum))
