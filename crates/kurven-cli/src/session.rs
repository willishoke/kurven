//! `kurven-cli session REQUESTS [-o out.svg] [--state S.json] [--edited E.json]
//! [--stage NAME DIR]... [--save NAME DIR] [--replies DIR]`: the page's protocol
//! run natively. Each line of REQUESTS is one request as
//! the page sends it to its Worker (`web/app/src/engine/protocol.ts`), run in
//! order through the same `Session` the wasm module wraps; a refusal is
//! printed and the run goes on, as the page goes on. Then the judged strokes
//! of the document the requests left, as the page's Export writes them
//! (`{"kind": "bake", "svg": true}`), to `-o`; the state and what is edited
//! as JSON, when asked. `--stage` stages a bundle directory's files under
//! NAME first, as the page stages a bundle it opens (an `open` names it);
//! `--save` writes what the page's Save bundle saves (`{"kind": "save"}`)
//! into DIR, its files at their paths; `--replies` writes every reply, its
//! JSON and its arrays (`replies`), as `crates/kurven-wasm/requests.mjs`
//! writes the wasm module's, which the check requires to be the same bytes.
//! The browser lane holds the page's exported SVG to
//! this, byte for byte (crates/FINDINGS.md, Phase 8).

use kurven_core::json::{self, JsonValue};
use kurven_session::protocol::Session;

/// Run the requests, then write what was asked for.
pub fn session_command(args: &[String]) -> Result<(), String> {
    let mut requests = None;
    let (mut svg, mut state, mut edited) = (None, None, None);
    let mut staged: Vec<(String, String)> = Vec::new();
    let mut save: Option<(String, String)> = None;
    let mut replies: Option<String> = None;
    let mut words = args.iter();
    while let Some(word) = words.next() {
        let mut value = |flag: &str| words.next().cloned().ok_or_else(|| format!("{flag} wants a path"));
        match word.as_str() {
            "-o" => svg = Some(value("-o")?),
            "--state" => state = Some(value("--state")?),
            "--edited" => edited = Some(value("--edited")?),
            "--stage" => {
                let name = value("--stage")?;
                staged.push((name, value("--stage")?));
            }
            "--replies" => replies = Some(value("--replies")?),
            "--save" => {
                let name = value("--save")?;
                save = Some((name, value("--save")?));
            }
            flag if flag.starts_with('-') => return Err(format!("session takes no {flag}")),
            path if requests.is_none() => requests = Some(path.to_string()),
            extra => return Err(format!("session takes one file of requests, not also {extra}")),
        }
    }
    let path = requests.ok_or("session wants a file of requests, one JSON object a line")?;
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
    let mut session = Session::new();
    for (name, dir) in &staged {
        stage_directory(&mut session, name, std::path::Path::new(dir), "")?;
    }
    if let Some(dir) = &replies {
        std::fs::create_dir_all(dir).map_err(|e| format!("{dir}: {e}"))?;
    }
    // A line starting with `#` is a comment (a fixture says what it stages).
    for (n, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty() && !l.starts_with('#')) {
        let answer = session.call(line);
        if let Some(dir) = &replies {
            write_reply(dir, n + 1, &answer)?;
        }
        let reply = json::parse(&answer.json).map_err(|e| format!("{path}:{}: {e}", n + 1))?;
        if reply.get("ok").and_then(JsonValue::as_bool) != Some(true) {
            let error = reply.get("error").and_then(JsonValue::as_str).unwrap_or("refused");
            eprintln!("kurven-cli: {path}:{}: refused: {error}", n + 1);
        }
    }
    if let Some(out) = state {
        let reply = json::parse(&session.call(r#"{"kind":"state"}"#).json).map_err(|e| e.to_string())?;
        let s = reply.get("state").cloned().unwrap_or(JsonValue::Null);
        std::fs::write(&out, s.canonical()).map_err(|e| format!("{out}: {e}"))?;
    }
    if let Some(out) = edited {
        let reply = json::parse(&session.call(r#"{"kind":"edited"}"#).json).map_err(|e| e.to_string())?;
        let e = reply.get("edited").cloned().unwrap_or(JsonValue::Null);
        std::fs::write(&out, e.canonical()).map_err(|e| format!("{out}: {e}"))?;
    }
    if let Some((name, dir)) = save {
        let request = JsonValue::Object(
            [("kind".to_string(), json::string("save")), ("name".to_string(), json::string(&name))]
                .into_iter()
                .collect(),
        );
        let reply = session.call(&request.canonical());
        let parsed = json::parse(&reply.json).map_err(|e| e.to_string())?;
        if parsed.get("ok").and_then(JsonValue::as_bool) != Some(true) {
            let error = parsed.get("error").and_then(JsonValue::as_str).unwrap_or("refused");
            return Err(format!("save: {error}"));
        }
        for (array, bytes) in reply.arrays {
            let Some(path) = array.strip_prefix("file:") else { continue };
            let out = std::path::Path::new(&dir).join(path);
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
            }
            std::fs::write(&out, bytes.into_bytes()).map_err(|e| format!("{}: {e}", out.display()))?;
        }
    }
    if let Some(out) = svg {
        let reply = session.call(r#"{"kind":"bake","svg":true}"#);
        let parsed = json::parse(&reply.json).map_err(|e| e.to_string())?;
        if parsed.get("ok").and_then(JsonValue::as_bool) != Some(true) {
            let error = parsed.get("error").and_then(JsonValue::as_str).unwrap_or("refused");
            return Err(format!("bake: {error}"));
        }
        let (_, bytes) = reply.arrays.into_iter().find(|(name, _)| name == "svg").ok_or("the bake wrote no SVG")?;
        std::fs::write(&out, bytes.into_bytes()).map_err(|e| format!("{out}: {e}"))?;
    }
    Ok(())
}

/// Stage every file under `dir` as the bundle `name`, at its path in it.
fn stage_directory(session: &mut Session, name: &str, dir: &std::path::Path, prefix: &str) -> Result<(), String> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .collect::<Result<_, _>>()
        .map_err(|e| format!("{}: {e}", dir.display()))?;
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let path = entry.path();
        let file = entry.file_name().to_string_lossy().into_owned();
        let at = if prefix.is_empty() { file } else { format!("{prefix}/{file}") };
        if path.is_dir() {
            stage_directory(session, name, &path, &at)?;
        } else {
            let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            session.stage(name, &at, bytes);
        }
    }
    Ok(())
}

/// A reply as `--replies` writes it: `NNNN.json`, and each array as
/// `NNNN.<name>.bin`, its name's characters outside `[A-Za-z0-9._-]` as `_`.
fn write_reply(dir: &str, line: usize, reply: &kurven_session::protocol::Reply) -> Result<(), String> {
    let base = std::path::Path::new(dir);
    let write = |name: String, bytes: &[u8]| {
        std::fs::write(base.join(&name), bytes).map_err(|e| format!("{}: {e}", base.join(&name).display()))
    };
    write(format!("{line:04}.json"), reply.json.as_bytes())?;
    for (name, array) in &reply.arrays {
        let safe: String =
            name.chars().map(|c| if c.is_ascii_alphanumeric() || "._-".contains(c) { c } else { '_' }).collect();
        write(format!("{line:04}.{safe}.bin"), array.bytes())?;
    }
    Ok(())
}
