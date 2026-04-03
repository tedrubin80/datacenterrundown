"""Minimal static file server for Railway deployment."""

import http.server
import os

PORT = int(os.environ.get("PORT", 8080))

handler = http.server.SimpleHTTPRequestHandler
handler.directory = os.path.dirname(os.path.abspath(__file__))

with http.server.HTTPServer(("0.0.0.0", PORT), handler) as httpd:
    print(f"Serving on port {PORT}")
    httpd.serve_forever()
