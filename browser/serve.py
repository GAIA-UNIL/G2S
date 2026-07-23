#!/usr/bin/env python3
"""Serve the G2S browser engine locally with WebAssembly thread headers."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class IsolatedRequestHandler(SimpleHTTPRequestHandler):
    """Static-file handler that enables SharedArrayBuffer on supported browsers."""

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the G2S browser page with COOP/COEP headers.",
    )
    parser.add_argument("--bind", default="127.0.0.1", help="address to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")
    arguments = parser.parse_args()
    if not 1 <= arguments.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    browser_directory = Path(__file__).resolve().parent
    handler = partial(IsolatedRequestHandler, directory=str(browser_directory))
    server = ThreadingHTTPServer((arguments.bind, arguments.port), handler)
    print(f"Serving G2S at http://localhost:{arguments.port}/")
    print("COOP/COEP enabled: the pthread build can use SharedArrayBuffer.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
