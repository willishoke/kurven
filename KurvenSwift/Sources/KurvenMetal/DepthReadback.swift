import Foundation
import Metal
import KurvenCore

/// Turning a rendered texture back into a value.
///
/// The bake needs this anyway, and having it means the absent GPU frame-capture
/// viewer costs nothing: a test can assert on a `DepthImage`, and
/// `kurven-cli inspect` can dump one, which is more of what a frame capture is
/// actually used for than the capture UI provides.
public enum DepthReadback {
    public static func read(_ texture: MTLTexture, frame: DepthFrame,
                            empty sentinel: Float) -> DepthImage {
        var values = [Float](repeating: sentinel, count: frame.rows * frame.cols)
        values.withUnsafeMutableBytes { bytes in
            texture.getBytes(bytes.baseAddress!,
                             bytesPerRow: frame.cols * MemoryLayout<Float>.stride,
                             from: MTLRegionMake2D(0, 0, frame.cols, frame.rows),
                             mipmapLevel: 0)
        }
        // Map the clear sentinel back to -infinity, so "nothing here" compares
        // the way the Python fill does and the visibility test needs no
        // separate coverage mask.
        let threshold = sentinel + 1e20
        for i in values.indices where values[i] <= threshold {
            values[i] = -.infinity
        }
        return DepthImage(frame: frame, values: values)
    }
}
