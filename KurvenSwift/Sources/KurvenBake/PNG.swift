import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import Metal
import KurvenCore

/// Writing a rendered texture to a PNG.
///
/// Not a feature of the app -- it is how the preview gets tested. A picture the
/// GUI draws can only be checked by looking at it; the same picture rendered
/// offscreen and written to a file can be diffed against the plate, which is
/// what makes "the preview looks like the bake" a claim rather than an
/// impression.
public enum PNG {
    public enum Error: Swift.Error, CustomStringConvertible {
        case unsupportedFormat(MTLPixelFormat)
        case encodingFailed(URL)

        public var description: String {
            switch self {
            case .unsupportedFormat(let f): "png: cannot write a \(f) texture"
            case .encodingFailed(let u): "png: could not encode \(u.lastPathComponent)"
            }
        }
    }

    public static func write(_ texture: MTLTexture, to url: URL) throws {
        guard texture.pixelFormat == .bgra8Unorm else {
            throw Error.unsupportedFormat(texture.pixelFormat)
        }
        let w = texture.width, h = texture.height
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        bytes.withUnsafeMutableBytes { p in
            texture.getBytes(p.baseAddress!, bytesPerRow: w * 4,
                             from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
        }
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue
                                | CGBitmapInfo.byteOrder32Little.rawValue)
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let image = CGImage(width: w, height: h, bitsPerComponent: 8,
                                  bitsPerPixel: 32, bytesPerRow: w * 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: info, provider: provider,
                                  decode: nil, shouldInterpolate: false,
                                  intent: .defaultIntent),
              let dest = CGImageDestinationCreateWithURL(
                  url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw Error.encodingFailed(url)
        }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw Error.encodingFailed(url) }
    }
}
