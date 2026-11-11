import os
import sys
import streamlit.web.cli as stcli


def resolve_path(path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, path)
    return os.path.join(os.path.abspath("."), path)


if __name__ == "__main__":
    if hasattr(sys, "_MEIPASS"):
        os.chdir(sys._MEIPASS)
    app_path = resolve_path("app.py")
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--global.developmentMode=false",
        "--server.headless=true",
    ]
    sys.exit(stcli.main())
