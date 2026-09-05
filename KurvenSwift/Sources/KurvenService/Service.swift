import Foundation
import KurvenCore

/// The Python half of the pipeline, over a pipe.
///
/// The contract is still the file: a request asks for a bundle and the reply
/// says where it is. That keeps every test able to run against a bundle on disk
/// with no live process, and it keeps the schema from drifting in two
/// directions -- there is one `.kurven` format and the socket is a way of
/// asking for one.
///
/// What the service adds is the thing a frozen bundle cannot do: change the
/// domain, or the resolution, or the function, and get a new landscape. That is
/// the only reason Python is still here, and it is exactly the part that was
/// always offline.
public final class Service: @unchecked Sendable {
    /// How to start the server.
    public struct Command: Sendable, Equatable {
        public var executable: URL
        public var arguments: [String]
        public var directory: URL?

        public init(executable: URL, arguments: [String], directory: URL? = nil) {
            self.executable = executable
            self.arguments = arguments
            self.directory = directory
        }

        public var display: String {
            ([executable.path] + arguments).joined(separator: " ")
        }

        /// Find a repository near `url` and the interpreter that goes with it.
        ///
        /// Walks up from a bundle's own location looking for `kurven/serve.py`,
        /// then prefers the virtualenv beside it. A bundle is usually written
        /// next to the tree that made it, and when it is not, the environment
        /// says so: `KURVEN_REPO` and `KURVEN_PYTHON` override, in that order of
        /// desperation.
        public static func autodetect(near url: URL?) -> Command? {
            let environment = ProcessInfo.processInfo.environment
            var roots: [URL] = []
            if let repo = environment["KURVEN_REPO"] {
                roots.append(URL(fileURLWithPath: repo))
            }
            var walk = url?.resolvingSymlinksInPath()
            while let current = walk, current.path != "/" {
                roots.append(current)
                walk = current.deletingLastPathComponent()
            }
            roots.append(URL(fileURLWithPath: FileManager.default.currentDirectoryPath))
            for root in roots {
                let serve = root.appendingPathComponent("kurven/serve.py")
                guard FileManager.default.fileExists(atPath: serve.path) else { continue }
                let python = environment["KURVEN_PYTHON"].map { URL(fileURLWithPath: $0) }
                    ?? [root.appendingPathComponent(".venv/bin/python"),
                        URL(fileURLWithPath: "/usr/bin/python3")]
                        .first { FileManager.default.isExecutableFile(atPath: $0.path) }
                guard let python else { continue }
                return Command(executable: python,
                               arguments: ["-m", "kurven.serve"],
                               directory: root)
            }
            return nil
        }
    }

    public enum Failure: Error, CustomStringConvertible {
        case cannotStart(String)
        case notRunning
        case remote(kind: String, message: String)
        case malformedReply(String)

        public var description: String {
            switch self {
            case .cannotStart(let m): "service: could not start (\(m))"
            case .notRunning: "service: the server is not running"
            case .remote(let kind, let message): "service: \(kind): \(message)"
            case .malformedReply(let m): "service: malformed reply (\(m))"
            }
        }
    }

    private let queue = DispatchQueue(label: "world.kurven.service")
    private let process = Process()
    private let toChild = Pipe()
    private let fromChild = Pipe()
    private var pending = Data()
    private var nextID = 1

    public let command: Command

    public init(command: Command) throws {
        self.command = command
        process.executableURL = command.executable
        process.arguments = command.arguments
        process.currentDirectoryURL = command.directory
        process.standardInput = toChild
        process.standardOutput = fromChild
        // The server's own chatter goes to the terminal, not into the stream:
        // anything on stdout that is not a reply would corrupt it.
        process.standardError = FileHandle.standardError
        do { try process.run() } catch {
            throw Failure.cannotStart("\(error)")
        }
    }

    public var isRunning: Bool { process.isRunning }

    public func stop() {
        queue.sync {
            guard process.isRunning else { return }
            try? toChild.fileHandleForWriting.close()
            process.terminate()
            process.waitUntilExit()
        }
    }

    /// One request, one reply.
    ///
    /// Serialized on a private queue rather than an actor because the read is a
    /// blocking one, and blocking a cooperative executor to wait for a
    /// subprocess to finish sampling is how a UI stops repainting.
    public func request(_ method: String, _ params: JSONValue = .object([:])) async throws
        -> JSONValue {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do { continuation.resume(returning: try self.exchange(method, params)) }
                catch { continuation.resume(throwing: error) }
            }
        }
    }

    private func exchange(_ method: String, _ params: JSONValue) throws -> JSONValue {
        guard process.isRunning else { throw Failure.notRunning }
        let id = nextID
        nextID += 1
        let request = JSONValue.object(["id": .int(id), "method": .string(method),
                                        "params": params])
        try toChild.fileHandleForWriting.write(contentsOf: Data((request.canonical + "\n").utf8))

        while true {
            guard let line = try readLine() else { throw Failure.notRunning }
            let reply = try JSONValue.parse(line)
            let object = try reply.object("reply")
            // Replies are tagged, so a stray one from a cancelled request is
            // skipped rather than mistaken for this one's answer.
            if case .some(.int(let replyID)) = object["id"], replyID != id { continue }
            if let error = object["error"] {
                let e = try error.object("error")
                throw Failure.remote(kind: try e.string("kind", "error"),
                                     message: try e.string("message", "error"))
            }
            guard let result = object["result"] else {
                throw Failure.malformedReply("no result and no error")
            }
            return result
        }
    }

    private func readLine() throws -> Data? {
        while true {
            if let newline = pending.firstIndex(of: UInt8(ascii: "\n")) {
                let line = pending[pending.startIndex..<newline]
                pending.removeSubrange(pending.startIndex...newline)
                return Data(line)
            }
            let chunk = fromChild.fileHandleForReading.availableData
            if chunk.isEmpty { return pending.isEmpty ? nil : nil }
            pending.append(chunk)
        }
    }
}
