"""
Web-based Training Dashboard Server

Replaces terminal LiveDisplay with browser-based UI.
- No layout shift issues (HTML/CSS is deterministic)
- Works in SSH, Docker, remote sessions
- Access at http://localhost:5050 during training

Usage:
    from utils.dashboard_server import start_dashboard_server
    start_dashboard_server(port=5050, open_browser=True)
"""

import os
import re
import threading
from dataclasses import asdict, fields
from typing import Optional

# Regex to strip ANSI escape codes (e.g., \033[33m, \033[0m)
_ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-9;]*m')

# Flask import with fallback
try:
    from flask import Flask, render_template, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = None

# Get the directory where this file is located
_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))

# Create Flask app with correct template/static paths
if FLASK_AVAILABLE:
    app = Flask(
        __name__,
        template_folder=os.path.join(_UTILS_DIR, 'templates'),
        static_folder=os.path.join(_UTILS_DIR, 'static')
    )
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # No caching for development
else:
    app = None

_server_thread: Optional[threading.Thread] = None
_server_started = False
_server_port = 5050


def _get_display_state_dict() -> dict:
    """Get the current display state as a dictionary."""
    try:
        from utils.live_display import get_display
        display = get_display()
        if display and display.state:
            # Convert dataclass to dict
            state_dict = asdict(display.state)
            # Ensure event_log is a list (deque -> list) and strip ANSI codes
            if 'event_log' in state_dict:
                raw_log = list(state_dict['event_log']) if state_dict['event_log'] else []
                # Strip ANSI escape codes for web display
                state_dict['event_log'] = [_ANSI_ESCAPE_RE.sub('', entry) for entry in raw_log]
            return state_dict
    except Exception as e:
        print(f"[Dashboard] Error getting display state: {e}")
    return {}


def _get_recent_logs(n_lines: int = 50) -> list:
    """Get recent log entries from the log file."""
    try:
        from utils.live_display import get_display
        display = get_display()
        if display and hasattr(display, 'log_file') and display.log_file:
            if os.path.exists(display.log_file):
                with open(display.log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[-n_lines:]
                    return [line.rstrip() for line in lines]
    except Exception:
        pass
    return []


if FLASK_AVAILABLE and app:
    @app.route('/')
    def dashboard():
        """Serve the main dashboard page."""
        return render_template('dashboard.html')

    @app.route('/api/state')
    def get_state():
        """Return current DisplayState as JSON."""
        state_dict = _get_display_state_dict()
        return jsonify(state_dict)

    @app.route('/api/logs')
    def get_logs():
        """Return recent log entries."""
        logs = _get_recent_logs(50)
        return jsonify({'logs': logs})

    @app.route('/api/health')
    def health():
        """Health check endpoint."""
        return jsonify({'status': 'ok', 'port': _server_port})


def start_dashboard_server(port: int = 5050, open_browser: bool = True) -> bool:
    """
    Start the dashboard server in a background thread.

    Args:
        port: Port to run the server on (default: 5050)
        open_browser: Whether to automatically open the browser (default: True)

    Returns:
        True if server started successfully, False otherwise
    """
    global _server_thread, _server_started, _server_port

    if not FLASK_AVAILABLE:
        print("[Dashboard] Flask not installed. Run: pip install flask>=2.3.0")
        return False

    if _server_started:
        print(f"[Dashboard] Already running at http://localhost:{_server_port}")
        return True

    _server_port = port

    def run_server():
        """Run Flask server in thread."""
        import logging
        # Suppress Flask's startup messages
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        # Also suppress Flask's own logger
        app.logger.setLevel(logging.ERROR)

        try:
            app.run(
                host='0.0.0.0',
                port=port,
                debug=False,
                use_reloader=False,
                threaded=True
            )
        except Exception as e:
            print(f"[Dashboard] Server error: {e}")

    _server_thread = threading.Thread(target=run_server, daemon=True)
    _server_thread.start()
    _server_started = True

    print(f"📊 Dashboard available at: http://localhost:{port}")

    if open_browser:
        import webbrowser
        import time
        # Give server a moment to start
        time.sleep(0.5)
        try:
            webbrowser.open(f'http://localhost:{port}')
        except Exception:
            pass

    return True


def stop_dashboard_server():
    """
    Stop the dashboard server.

    Note: Flask doesn't have a clean shutdown in thread mode.
    The daemon thread will terminate when the main process exits.
    """
    global _server_started
    _server_started = False
    print("[Dashboard] Server will stop when main process exits")


def is_dashboard_running() -> bool:
    """Check if the dashboard server is running."""
    return _server_started


def get_dashboard_url() -> Optional[str]:
    """Get the dashboard URL if running."""
    if _server_started:
        return f"http://localhost:{_server_port}"
    return None


# For standalone testing
if __name__ == '__main__':
    if not FLASK_AVAILABLE:
        print("Flask not installed. Run: pip install flask>=2.3.0")
    else:
        print("Starting dashboard server for testing...")
        start_dashboard_server(port=5050, open_browser=True)
        # Keep main thread alive
        import time
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
