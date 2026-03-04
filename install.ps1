# CLI installer — downloads the latest release binary from GitHub.
# Configure $Repo and $Bin below. Everything else is derived automatically.
#
# Environment (upper-cased $Bin prefix):
#   <BIN>_VERSION      Override version (default: latest)
#   <BIN>_INSTALL_DIR  Override install directory (default: %LOCALAPPDATA%\<bin>)

$ErrorActionPreference = "Stop"
$InformationPreference = "Continue"
$Repo = "qntx/machi"
$Bin = "machi"

$BinUpper = $Bin.ToUpper()
$VerEnv = "${BinUpper}_VERSION"
$DirEnv = "${BinUpper}_INSTALL_DIR"

function Get-TargetArch {
    try {
        $a = [System.Reflection.Assembly]::LoadWithPartialName("System.Runtime.InteropServices.RuntimeInformation")
        switch ($a.GetType("System.Runtime.InteropServices.RuntimeInformation").GetProperty("OSArchitecture").GetValue($null).ToString()) {
            "X64" { return "x86_64-pc-windows-msvc" }
            "Arm64" { return "aarch64-pc-windows-msvc" }
        }
    }
    catch {}
    if ([Environment]::Is64BitOperatingSystem) { return "x86_64-pc-windows-msvc" }
    throw "32-bit Windows is not supported"
}

function Get-LatestVersion {
    $tag = (Invoke-RestMethod "https://api.github.com/repos/$Repo/releases/latest").tag_name
    if ($tag.StartsWith("v")) { $tag = $tag.Substring(1) }
    return $tag
}

function Add-ToUserPath($Dir) {
    $reg = 'registry::HKEY_CURRENT_USER\Environment'
    $current = (Get-Item -LiteralPath $reg).GetValue('Path', '', 'DoNotExpandEnvironmentNames') -split ';' -ne ''
    if ($Dir -in $current) { return }

    Set-ItemProperty -Type ExpandString -LiteralPath $reg Path ((@($Dir) + $current) -join ';')
    # Broadcast WM_SETTINGCHANGE so Explorer picks up the new PATH
    $k = "$Bin-" + [guid]::NewGuid().ToString()
    [Environment]::SetEnvironmentVariable($k, "1", "User")
    [Environment]::SetEnvironmentVariable($k, [NullString]::value, "User")

    Write-Information "  Added $Dir to PATH (restart your shell to apply)"
}

try {
    $target = Get-TargetArch
    $envVer = [Environment]::GetEnvironmentVariable($VerEnv)
    $envDir = [Environment]::GetEnvironmentVariable($DirEnv)
    $ver = if ($envVer) { $envVer } else { Get-LatestVersion }
    $dir = if ($envDir) { $envDir } else { Join-Path $env:LOCALAPPDATA $Bin }

    Write-Information "Installing $Bin v$ver ($target)"

    $url = "https://github.com/$Repo/releases/download/v$ver/$Bin-$ver-$target.zip"
    $tmp = New-Item -ItemType Directory -Path (Join-Path ([IO.Path]::GetTempPath()) ([Guid]::NewGuid()))

    try {
        Invoke-WebRequest -Uri $url -OutFile "$tmp\archive.zip" -UseBasicParsing
        Expand-Archive "$tmp\archive.zip" -DestinationPath $tmp -Force

        $null = New-Item -ItemType Directory -Force -Path $dir
        Copy-Item "$tmp\$Bin.exe" -Destination $dir -Force
        Write-Information "  -> $(Join-Path $dir "$Bin.exe")"
    }
    finally {
        Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }

    # Add to PATH if needed
    if ($env:GITHUB_PATH) {
        $dir | Out-File $env:GITHUB_PATH -Encoding utf8 -Append
    }
    elseif (-not ($env:Path -split ';' | Where-Object { $_ -eq $dir })) {
        Add-ToUserPath $dir
    }

    Write-Information "`n$Bin v$ver installed successfully!"
}
catch {
    Write-Error $_
    exit 1
}
