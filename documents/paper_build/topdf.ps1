param($src, $pdf)
$ErrorActionPreference = "Stop"
try {
  $word = New-Object -ComObject Word.Application
  $word.Visible = $false
  $word.DisplayAlerts = 0
  $doc = $word.Documents.Open($src, $false, $true)
  Write-Output ("pages: " + $doc.ComputeStatistics(2))
  $doc.ExportAsFixedFormat($pdf, 17)
  $doc.Close($false)
  $word.Quit()
  Write-Output "ok"
} catch { Write-Output ("ERROR: " + $_.Exception.Message); if ($word) { $word.Quit() } }
