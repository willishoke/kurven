#!/bin/bash
# Assemble Kurven.app around the KurvenApp binary.
#
# A bare SwiftPM executable has no bundle, and without one macOS gives it no
# Info.plist, so it cannot declare a document type, cannot be launched by the
# Finder with a file, and gets a generic menu bar. Xcode would assemble one; so
# does this, out of Command Line Tools, and the point of the design's SwiftPM-
# only commitment is that these stay the same arrangement whether or not Xcode
# is ever installed.
#
#   scripts/bundle-app.sh [--debug] [--output DIR]
#
# The exported type is a *package*: a .kurven bundle is a directory, and
# declaring it as one is what makes the Finder show it as a single document and
# the open panel offer it rather than descend into it.

set -euo pipefail

CONFIG=release
OUTPUT=build
while [ $# -gt 0 ]; do
    case "$1" in
        --debug) CONFIG=debug; shift ;;
        --output) OUTPUT="$2"; shift 2 ;;
        *) echo "usage: $0 [--debug] [--output DIR]" >&2; exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE="$ROOT/KurvenSwift"
APP="$ROOT/$OUTPUT/Kurven.app"
VERSION="0.1.0"

echo "building KurvenApp ($CONFIG)"
swift build -c "$CONFIG" --package-path "$PACKAGE" --product KurvenApp
BINARY="$(swift build -c "$CONFIG" --package-path "$PACKAGE" --show-bin-path)/KurvenApp"
[ -x "$BINARY" ] || { echo "no binary at $BINARY" >&2; exit 1; }

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BINARY" "$APP/Contents/MacOS/Kurven"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>                  <string>Kurven</string>
    <key>CFBundleDisplayName</key>           <string>Kurven</string>
    <key>CFBundleExecutable</key>            <string>Kurven</string>
    <key>CFBundleIdentifier</key>            <string>world.kurven.Kurven</string>
    <key>CFBundlePackageType</key>           <string>APPL</string>
    <key>CFBundleShortVersionString</key>    <string>$VERSION</string>
    <key>CFBundleVersion</key>               <string>$VERSION</string>
    <key>LSMinimumSystemVersion</key>        <string>15.0</string>
    <key>NSHighResolutionCapable</key>       <true/>
    <key>NSPrincipalClass</key>              <string>NSApplication</string>

    <!-- The .kurven bundle, declared as a package so the Finder shows it as one
         document instead of a folder of npy files. -->
    <key>UTExportedTypeDeclarations</key>
    <array>
        <dict>
            <key>UTTypeIdentifier</key>      <string>world.kurven.bundle</string>
            <key>UTTypeDescription</key>     <string>Kurven Landscape Bundle</string>
            <key>UTTypeConformsTo</key>
            <array>
                <string>com.apple.package</string>
                <string>public.composite-content</string>
            </array>
            <key>UTTypeTagSpecification</key>
            <dict>
                <key>public.filename-extension</key>
                <array><string>kurven</string></array>
            </dict>
        </dict>
    </array>
    <key>CFBundleDocumentTypes</key>
    <array>
        <dict>
            <key>CFBundleTypeName</key>      <string>Kurven Landscape Bundle</string>
            <key>CFBundleTypeRole</key>      <string>Viewer</string>
            <key>LSHandlerRank</key>         <string>Owner</string>
            <key>LSTypeIsPackage</key>       <true/>
            <key>LSItemContentTypes</key>
            <array><string>world.kurven.bundle</string></array>
        </dict>
    </array>
</dict>
</plist>
PLIST

# Ad hoc: enough for the app to launch and for Metal to hand it a device.
# Distribution would want a Developer ID, which is a separate conversation.
codesign --force --sign - --timestamp=none "$APP" >/dev/null 2>&1 \
    || echo "warning: codesign failed; the app may still run locally" >&2

# The Finder caches document types per app path; registering makes a fresh
# bundle's .kurven association take effect without a logout.
LSREGISTER=/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister
[ -x "$LSREGISTER" ] && "$LSREGISTER" -f "$APP" || true

echo "wrote $APP"
echo "  open -a '$APP' path/to/bundle.kurven"
