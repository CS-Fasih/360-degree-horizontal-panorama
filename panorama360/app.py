"""Application entry point."""

from __future__ import annotations

import os
import sys

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from .gui import MainWindow


def main() -> int:
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    application = QApplication(sys.argv)
    application.setApplicationName("Panorama 360")
    application.setOrganizationName("CS-Fasih")
    application.setStyle("Fusion")
    application.setFont(QFont("Sans Serif", 10))
    window = MainWindow()
    window.show()
    return application.exec()
