#!/bin/sh
# CLI installer — downloads the latest release binary from GitHub.
# Configure REPO and BIN below. Everything else is derived automatically.
#
# Environment (upper-cased BIN prefix):
#   <BIN>_VERSION      Override version (default: latest)
#   <BIN>_INSTALL_DIR  Override install directory (default: ~/.local/bin)

set -eu

REPO="qntx/machi"
BIN="machi"

BIN_UPPER=$(echo "$BIN" | tr '[:lower:]' '[:upper:]')

say() { printf '\033[1m%s\033[0m\n' "$*"; }
err() { printf '\033[31merror\033[0m: %s\n' "$*" >&2; exit 1; }

fetch() {
    if command -v curl >/dev/null 2>&1; then curl -fsSL "$1"
    elif command -v wget >/dev/null 2>&1; then wget -qO- "$1"
    else err "curl or wget is required"; fi
}

download() {
    if command -v curl >/dev/null 2>&1; then curl -fsSL -o "$2" "$1"
    elif command -v wget >/dev/null 2>&1; then wget -q -O "$2" "$1"
    else err "curl or wget is required"; fi
}

detect_target() {
    os=$(uname -s)
    arch=$(uname -m)

    case "$os" in
        Linux)  os="unknown-linux-gnu" ;;
        Darwin)
            os="apple-darwin"
            # Detect Apple Silicon behind Rosetta 2
            [ "$arch" = x86_64 ] && sysctl -n hw.optional.arm64 2>/dev/null | grep -q 1 && arch=aarch64
            ;;
        *) err "unsupported OS: $os" ;;
    esac

    case "$arch" in
        x86_64 | amd64) arch=x86_64 ;;
        aarch64 | arm64) arch=aarch64 ;;
        *) err "unsupported architecture: $arch" ;;
    esac

    echo "${arch}-${os}"
}

latest_version() {
    tag=$(fetch "https://api.github.com/repos/$REPO/releases/latest" \
        | grep '"tag_name"' | head -1 | cut -d'"' -f4)
    [ -n "$tag" ] || err "failed to detect latest version"
    echo "${tag#v}"
}

main() {
    target=$(detect_target)
    eval "ver=\${${BIN_UPPER}_VERSION:-\$(latest_version)}"
    eval "dir=\${${BIN_UPPER}_INSTALL_DIR:-\$HOME/.local/bin}"

    say "Installing $BIN v$ver ($target)"

    url="https://github.com/$REPO/releases/download/v$ver/$BIN-$ver-$target.tar.gz"
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT

    download "$url" "$tmp/archive.tar.gz"
    tar xzf "$tmp/archive.tar.gz" -C "$tmp"

    mkdir -p "$dir"
    install -m 755 "$tmp/$BIN" "$dir/$BIN"

    say "  → $dir/$BIN"

    # Add to PATH if needed
    case ":${PATH}:" in *":$dir:"*) return ;; esac

    line="export PATH=\"$dir:\$PATH\""
    for rc in .zshrc .bashrc .bash_profile .profile; do
        [ -f "$HOME/$rc" ] || continue
        grep -qF "$dir" "$HOME/$rc" 2>/dev/null && return
        printf '\n%s\n' "$line" >> "$HOME/$rc"
        say "  Added $dir to PATH in ~/$rc (restart your shell to apply)"
        return
    done

    # No rc file found — create .profile
    printf '%s\n' "$line" > "$HOME/.profile"
    say "  Created ~/.profile with PATH entry (restart your shell to apply)"
}

main
