"""`python -m kurven.serve` — newline-delimited JSON-RPC on stdio.

The file is the protocol; this is a delivery mechanism for it. A request asks
for a bundle and the reply says where it is, because a bundle is a value with a
checksum that both sides can already read, and streaming a hundred megabytes of
npy through a pipe to avoid writing it down would be a strange way to save
effort. Everything the frontend does with the answer, it can also do with a
bundle someone handed it on disk.

    {"id": 1, "method": "describe"}
    {"id": 1, "result": {"examples": [...]}}

    {"id": 2, "method": "catalog"}
    {"id": 2, "result": {"presets": [...], "language": {...}}}

    {"id": 3, "method": "landscape", "params": {"expression": "1/gamma(z)",
                                                "domain": {...}, "resolution": 600,
                                                "output": "/tmp/l.kurven"}}
    {"id": 3, "result": {"path": "...", "bytes": 260000, "manifest": {...}}}

    {"id": 4, "method": "export", "params": {"example": "recip",
                                             "output": "/tmp/r.kurven",
                                             "arguments": {"res": 800}}}
    {"id": 2, "result": {"path": "...", "bytes": 5100000, "manifest": {...}}}

One request per line, one reply per line, replies tagged with the request's id.
Requests are served in order on one thread: sampling is CPU-bound and the client
that wants two landscapes at once can run two servers.

`catalog` and `landscape` are the other half: not "run this published plate"
but "sample this function over this rectangle". A landscape's bundle describes
every one of its layers rather than dumping any, so the client can move the cap,
the levels and the hatch spacing itself and only comes back here when the
function or the window changes -- which is the one thing that needs f evaluated
again. `validate` is the same parser without the sampling, for a text field.

`describe` reports each example's own argparse options -- name, type, default,
help -- so a client can build a form for a function it has never heard of, and
adding an option to an example makes it appear there without anyone editing the
client. That is the whole reason the examples were given a `parser()` separate
from `main()`.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

from kurven.export import EXAMPLES, export, load_example

PROTOCOL = 1


class RPCError(Exception):
    """An error with a kind the client can branch on, rather than a string it
    has to parse."""

    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind
        self.message = message


def describe_parser(parser):
    """An example's options as data.

    argparse already knows the name, type, default and help of everything an
    example accepts; this is that, in a shape a client can render.
    """
    out = []
    for action in parser._actions:
        if action.dest in ("help", "output_prefix"):
            continue
        kind = "flag" if action.nargs == 0 else {
            int: "int", float: "float", str: "string",
        }.get(action.type, "string")
        out.append({
            "name": action.dest,
            "kind": kind,
            "default": action.default,
            "help": action.help or "",
            "choices": list(action.choices) if action.choices else None,
        })
    return out


def to_argv(parser, arguments):
    """Turn `{"res": 800, "gpu": true}` into the argv the example expects.

    Going back through argparse rather than around it means the service accepts
    exactly what the command line accepts, including its validation, and an
    example cannot grow an option the service silently ignores.
    """
    by_dest = {a.dest: a for a in parser._actions}
    argv = []
    for name, value in (arguments or {}).items():
        action = by_dest.get(name)
        if action is None:
            raise RPCError("unknownArgument",
                           f"no option {name!r}; try 'describe'")
        flag = action.option_strings[-1] if action.option_strings else None
        if flag is None:
            raise RPCError("unknownArgument", f"{name!r} is positional")
        if action.nargs == 0:
            if value:
                argv.append(flag)
        else:
            argv += [flag, str(value)]
    return argv


# --------------------------------------------------------------------------
# methods
# --------------------------------------------------------------------------


def method_describe(_params):
    examples = []
    for name in sorted(EXAMPLES):
        try:
            module = load_example(name)
        except SystemExit as e:
            examples.append({"name": name, "available": False, "reason": str(e)})
            continue
        examples.append({
            "name": name,
            "available": True,
            "arguments": describe_parser(module.parser()),
        })
    return {"protocol": PROTOCOL, "examples": examples}


def method_export(params):
    name = params.get("example")
    if name not in EXAMPLES:
        raise RPCError("unknownExample",
                       f"no example {name!r}; have {sorted(EXAMPLES)}")
    output = params.get("output")
    if not output:
        raise RPCError("badRequest", "export needs an 'output' path")

    module = load_example(name)
    parser = module.parser()
    try:
        args = parser.parse_args(to_argv(parser, params.get("arguments")))
    except SystemExit:
        raise RPCError("badArguments", "the example rejected those arguments")
    args.chunk_count = int(params.get("chunkCount", 1))

    # An example prints as it works, and anything on stdout that is not a reply
    # corrupts the stream. It goes to stderr, where a client that wants progress
    # can read it.
    noise = io.StringIO()
    with redirect_stdout(noise):
        scene = module.build_scene(args, verbose=True)
        if params.get("derived"):
            import dataclasses
            if scene.perimeter is None:
                raise RPCError("notDerivable",
                               f"{name} has no perimeter to derive walls from")
            scene = dataclasses.replace(scene, walls=())
        manifest = export(scene, output, chunk_count=args.chunk_count,
                          phase=not params.get("noPhase", False),
                          wall_mesh=not params.get("derived", False),
                          derived=bool(params.get("derived", False)),
                          example=name)
    sys.stderr.write(noise.getvalue())
    sys.stderr.flush()

    path = Path(output)
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return {"path": str(path.resolve()), "bytes": total,
            "manifest": manifest.to_dict()}


def method_catalog(_params):
    """The menu, and the language behind it.

    `presets` is a list of good starting points; `language` is everything the
    expression field will accept -- each function's name, arity and aliases --
    so a client can offer them and reject a typo before paying for a round trip.
    Neither is written down in the client, because a function added here should
    appear there without anyone editing Swift.
    """
    from kurven.expr import LANGUAGE
    from kurven.landscape import (CATALOG, DEFAULT_RESOLUTION, PLATE_BUFFER,
                                  PLATE_MARGIN)

    return {"protocol": PROTOCOL,
            "presets": [p.to_dict() for p in CATALOG],
            "language": LANGUAGE,
            "defaults": {"resolution": DEFAULT_RESOLUTION,
                         "buffer": PLATE_BUFFER, "margin": PLATE_MARGIN}}


def method_validate(params):
    """Parse an expression and say what it means, without sampling anything.

    A parse is microseconds and a landscape is a second, so a field being typed
    into should not be asking for landscapes. The answer is the canonical
    spelling, which is also how a client learns that `1/Γ(z)` and `1 / gamma(z)`
    are the same landscape.
    """
    from kurven.expr import ExpressionError, canonical

    try:
        return {"expression": canonical(params.get("expression", ""))}
    except ExpressionError as e:
        raise RPCError("badExpression", str(e))


def method_landscape(params):
    """Sample a function over a rectangle and write the bundle for it.

    The request is a `kurven.landscape.Spec` -- an expression, a window, a
    resolution, and the styling the client has edited, if it has -- and the
    answer is a bundle every layer of which is a description. Nothing is
    contoured here: the consumer derives the ink from the grids, and that is
    what makes the cap, the levels and the hatch spacing editable without asking
    this process for anything.
    """
    import dataclasses

    from kurven.bundle import BundleError
    from kurven.expr import ExpressionError
    from kurven.landscape import Spec, build_scene

    output = params.get("output")
    if not output:
        raise RPCError("badRequest", "landscape needs an 'output' path")
    try:
        spec = Spec.from_dict(params)
    except (BundleError, KeyError, TypeError, ValueError) as e:
        raise RPCError("badRequest", str(e))

    chunks = int(params.get("chunkCount", 1))
    noise = io.StringIO()
    try:
        with redirect_stdout(noise):
            scene = build_scene(spec, verbose=True, geometry=False,
                                chunk_count=chunks)
            scene = dataclasses.replace(scene, walls=())
            manifest = export(scene, output, chunk_count=chunks,
                              phase=not params.get("noPhase", False),
                              wall_mesh=False, derived=True, example="function")
    except ExpressionError as e:
        raise RPCError("badExpression", str(e))
    sys.stderr.write(noise.getvalue())
    sys.stderr.flush()

    path = Path(output)
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return {"path": str(path.resolve()), "bytes": total,
            "manifest": manifest.to_dict()}


METHODS = {"describe": method_describe, "export": method_export,
           "catalog": method_catalog, "validate": method_validate,
           "landscape": method_landscape}


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------


def handle(request):
    method = METHODS.get(request.get("method"))
    if method is None:
        raise RPCError("unknownMethod",
                       f"no method {request.get('method')!r}; "
                       f"have {sorted(METHODS)}")
    return method(request.get("params") or {})


def serve(stdin=None, stdout=None):
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            reply = {"id": None, "error": {"kind": "badJSON", "message": str(e)}}
        else:
            rid = request.get("id")
            try:
                reply = {"id": rid, "result": handle(request)}
            except RPCError as e:
                reply = {"id": rid, "error": {"kind": e.kind, "message": e.message}}
            except Exception as e:                       # noqa: BLE001
                traceback.print_exc(file=sys.stderr)
                reply = {"id": rid, "error": {"kind": type(e).__name__,
                                              "message": str(e)}}
        try:
            stdout.write(json.dumps(reply, allow_nan=False) + "\n")
            stdout.flush()
        except BrokenPipeError:
            # The client left while we were answering, which is not an error:
            # a window that takes a screenshot and exits does exactly this, and
            # a server that treats it as a crash prints a traceback over the
            # output of whatever the client was actually doing.
            return


def _quiet_exit():
    """Let the interpreter shut down after the client has gone.

    Python flushes stdout on the way out, and if the read end is closed that
    raises a second `BrokenPipeError` in the teardown, where nothing can catch
    it -- so it prints "Exception ignored" and an exit code of 120. Pointing the
    descriptor at /dev/null first is the documented way to have nothing to
    flush (see the Python FAQ on BrokenPipeError).
    """
    try:
        sys.stdout.flush()
    except BrokenPipeError:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m kurven.serve", description=__doc__)
    ap.add_argument("--once", type=str, default=None,
                    help="handle a single request given as JSON and exit; for "
                         "testing the protocol without a client")
    args = ap.parse_args(argv)
    if args.once:
        serve(stdin=io.StringIO(args.once + "\n"))
        return
    serve()
    _quiet_exit()


if __name__ == "__main__":
    main()
