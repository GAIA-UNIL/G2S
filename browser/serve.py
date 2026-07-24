#!/usr/bin/env python3
"""Serve the G2S browser engine locally with WebAssembly thread headers."""

import argparse
import socket
import threading
import time
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


class IPv6ThreadingHTTPServer(ThreadingHTTPServer):
    """IPv6 loopback server used alongside the IPv4 localhost listener."""

    address_family = socket.AF_INET6


def localhost_servers(port: int, handler):
    """Create one IPv4 and one IPv6 loopback listener for localhost."""
    return [
        ThreadingHTTPServer(("127.0.0.1", port), handler),
        IPv6ThreadingHTTPServer(("::1", port), handler),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the G2S browser page with COOP/COEP headers.",
    )
    parser.add_argument("--bind", default="localhost", help="address to bind (default: both localhost address families)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")
    arguments = parser.parse_args()
    if not 1 <= arguments.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    browser_directory = Path(__file__).resolve().parent
    handler = partial(IsolatedRequestHandler, directory=str(browser_directory))
    if arguments.bind == "localhost":
        servers = localhost_servers(arguments.port, handler)
    elif ":" in arguments.bind:
        servers = [IPv6ThreadingHTTPServer((arguments.bind, arguments.port), handler)]
    else:
        servers = [ThreadingHTTPServer((arguments.bind, arguments.port), handler)]
    threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in servers]
    print(f"Serving G2S at http://localhost:{arguments.port}/")
    print("COOP/COEP enabled: the pthread build can use SharedArrayBuffer.")
    try:
        for thread in threads:
            thread.start()
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=2)


if __name__ == "__main__":
    main()
