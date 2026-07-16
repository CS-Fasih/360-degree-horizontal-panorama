"""PySide6 desktop interface for the panorama engine."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageOps
from PySide6.QtCore import QObject, QRunnable, QSize, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStyle,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .image_io import SUPPORTED_EXTENSIONS, save_rgb
from .models import AnalysisResult, PanoramaError, StitchResult
from .ordering import SequenceAnalyzer
from .stitcher import CylindricalStitcher


class _TaskSignals(QObject):
    progress = Signal(int, str)
    succeeded = Signal(object)
    failed = Signal(str, str, str)


class _Task(QRunnable):
    def __init__(self, operation: Callable[[Callable[[int, str], None]], object]) -> None:
        super().__init__()
        self.operation = operation
        self.signals = _TaskSignals()

    def run(self) -> None:
        try:
            result = self.operation(self.signals.progress.emit)
        except PanoramaError as exc:
            self.signals.failed.emit(exc.title, exc.user_message(), "")
        except Exception as exc:  # pragma: no cover - final GUI safety boundary
            self.signals.failed.emit(
                "Unexpected processing error",
                f"The operation could not be completed.\n\n{exc}",
                traceback.format_exc(),
            )
        else:
            self.signals.succeeded.emit(result)


class PanoramaViewer(QGraphicsView):
    """Fit-to-window result preview with optional wheel zoom."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._pixmap_item = QGraphicsPixmapItem()
        self.scene().addItem(self._pixmap_item)
        self._has_image = False
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setBackgroundBrush(QColor("#111827"))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

    def set_rgb(self, image_rgb: np.ndarray) -> None:
        height, width = image_rgb.shape[:2]
        qimage = QImage(
            image_rgb.data,
            width,
            height,
            int(image_rgb.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
        self._pixmap_item.setPixmap(QPixmap.fromImage(qimage))
        self.scene().setSceneRect(self._pixmap_item.boundingRect())
        self._has_image = True
        self.fit_to_window()

    def clear_image(self) -> None:
        self._pixmap_item.setPixmap(QPixmap())
        self._has_image = False

    def fit_to_window(self) -> None:
        if self._has_image:
            self.resetTransform()
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self.fit_to_window()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt API
        if self._has_image and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
            self.scale(factor, factor)
            event.accept()
            return
        super().wheelEvent(event)


class PhotoList(QListWidget):
    order_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setFlow(QListWidget.Flow.LeftToRight)
        self.setWrapping(True)
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setMovement(QListWidget.Movement.Snap)
        self.setIconSize(QSize(150, 92))
        self.setGridSize(QSize(174, 136))
        self.setSpacing(6)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setMinimumHeight(175)
        self.model().rowsMoved.connect(lambda *_args: self.order_changed.emit())

    def paths(self) -> list[Path]:
        return [Path(self.item(row).data(Qt.ItemDataRole.UserRole)) for row in range(self.count())]

    def renumber(self) -> None:
        for row in range(self.count()):
            item = self.item(row)
            path = Path(item.data(Qt.ItemDataRole.UserRole))
            item.setText(f"{row + 1}. {path.name}")


class MainWindow(QMainWindow):
    """One-window guided panorama workflow."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Panorama 360")
        self.resize(1180, 820)
        self.setMinimumSize(880, 650)
        self._thread_pool = QThreadPool.globalInstance()
        self._thread_pool.setMaxThreadCount(1)
        self._active_task: _Task | None = None
        self._analysis: AnalysisResult | None = None
        self._result: StitchResult | None = None
        self._busy = False
        self._applying_order = False

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(14)

        title = QLabel("Create a seamless 360° panorama")
        title.setObjectName("title")
        subtitle = QLabel(
            "Select one complete horizontal rotation. Photos may be out of order, but each should overlap its neighbours."
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        outer.addWidget(title)
        outer.addWidget(subtitle)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter, 1)

        source_panel = QFrame()
        source_panel.setObjectName("panel")
        source_layout = QVBoxLayout(source_panel)
        source_layout.setContentsMargins(16, 14, 16, 14)
        source_layout.setSpacing(10)

        source_header = QHBoxLayout()
        source_title = QLabel("Photos")
        source_title.setObjectName("sectionTitle")
        source_header.addWidget(source_title)
        source_header.addStretch()
        self.select_button = QPushButton("Select photos…")
        self.select_button.setObjectName("secondaryButton")
        self.remove_button = QPushButton("Remove selected")
        self.remove_button.setObjectName("secondaryButton")
        source_header.addWidget(self.remove_button)
        source_header.addWidget(self.select_button)
        source_layout.addLayout(source_header)

        hint = QLabel("Drag thumbnails to correct the order manually. The last photo must overlap the first.")
        hint.setObjectName("hint")
        source_layout.addWidget(hint)
        self.photo_list = PhotoList()
        source_layout.addWidget(self.photo_list)

        action_row = QHBoxLayout()
        self.analyze_button = QPushButton("Analyze & auto-order")
        self.analyze_button.setObjectName("secondaryButton")
        self.create_button = QPushButton("Create Panorama")
        self.create_button.setObjectName("primaryButton")
        self.save_button = QPushButton("Save panorama…")
        self.save_button.setObjectName("secondaryButton")
        self.analyze_button.setEnabled(False)
        self.create_button.setEnabled(False)
        self.save_button.setEnabled(False)
        action_row.addWidget(self.analyze_button)
        action_row.addStretch()
        action_row.addWidget(self.create_button)
        action_row.addWidget(self.save_button)
        source_layout.addLayout(action_row)

        self.status_label = QLabel("Select at least three photos to begin.")
        self.status_label.setObjectName("status")
        self.status_label.setWordWrap(True)
        source_layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.hide()
        source_layout.addWidget(self.progress_bar)
        splitter.addWidget(source_panel)

        result_panel = QFrame()
        result_panel.setObjectName("panel")
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(16, 14, 16, 14)
        preview_header = QHBoxLayout()
        preview_title = QLabel("Panorama preview")
        preview_title.setObjectName("sectionTitle")
        self.result_info = QLabel("No panorama created yet")
        self.result_info.setObjectName("hint")
        preview_header.addWidget(preview_title)
        preview_header.addStretch()
        preview_header.addWidget(self.result_info)
        result_layout.addLayout(preview_header)
        self.viewer = PanoramaViewer()
        self.viewer.setMinimumHeight(250)
        result_layout.addWidget(self.viewer, 1)
        preview_hint = QLabel("Hold Ctrl and use the mouse wheel to zoom; drag to pan.")
        preview_hint.setObjectName("hint")
        result_layout.addWidget(preview_hint)
        splitter.addWidget(result_panel)
        splitter.setSizes([330, 390])

        self.select_button.clicked.connect(self._select_photos)
        self.remove_button.clicked.connect(self._remove_selected)
        self.analyze_button.clicked.connect(self._start_analysis)
        self.create_button.clicked.connect(self._start_stitching)
        self.save_button.clicked.connect(self._save_result)
        self.photo_list.order_changed.connect(self._manual_order_changed)
        self._apply_style()

    def _select_photos(self) -> None:
        patterns = " ".join(f"*{suffix}" for suffix in sorted(SUPPORTED_EXTENSIONS))
        filenames, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            "Select overlapping panorama photos",
            str(Path.home()),
            f"Photos ({patterns});;All files (*)",
        )
        if not filenames:
            return
        self._set_photos([Path(filename) for filename in filenames])
        if self.photo_list.count() >= 3:
            self._start_analysis()

    def _set_photos(self, paths: list[Path]) -> None:
        self._applying_order = True
        try:
            self.photo_list.clear()
            for index, path in enumerate(paths):
                item = QListWidgetItem(f"{index + 1}. {path.name}")
                item.setData(Qt.ItemDataRole.UserRole, str(path))
                item.setToolTip(str(path))
                item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
                item.setIcon(self._thumbnail(path))
                item.setSizeHint(QSize(170, 132))
                self.photo_list.addItem(item)
        finally:
            self._applying_order = False
        self._analysis = None
        self._result = None
        self.viewer.clear_image()
        self.save_button.setEnabled(False)
        enough = len(paths) >= 3
        self.analyze_button.setEnabled(enough)
        self.create_button.setEnabled(False)
        self.status_label.setText(
            f"{len(paths)} photos selected. Analyzing overlap…" if enough else "Select at least three photos to begin."
        )

    @staticmethod
    def _thumbnail(path: Path) -> QIcon:
        try:
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                image.thumbnail((300, 184), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (300, 184), (31, 41, 55))
                offset = ((300 - image.width) // 2, (184 - image.height) // 2)
                canvas.paste(image, offset)
                array = np.asarray(canvas, dtype=np.uint8)
                qimage = QImage(
                    array.data,
                    array.shape[1],
                    array.shape[0],
                    int(array.strides[0]),
                    QImage.Format.Format_RGB888,
                ).copy()
                return QIcon(QPixmap.fromImage(qimage))
        except Exception:
            return QApplication.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon)

    def _remove_selected(self) -> None:
        if self._busy:
            return
        for item in self.photo_list.selectedItems():
            self.photo_list.takeItem(self.photo_list.row(item))
        self.photo_list.renumber()
        self._analysis = None
        self._result = None
        self.viewer.clear_image()
        self.save_button.setEnabled(False)
        enough = self.photo_list.count() >= 3
        self.analyze_button.setEnabled(enough)
        self.create_button.setEnabled(False)
        self.status_label.setText(
            "Photo selection changed. Analyze the sequence again."
            if enough
            else "Select at least three photos to begin."
        )

    def _start_analysis(self) -> None:
        paths = self.photo_list.paths()
        if len(paths) < 3 or self._busy:
            return
        analyzer = SequenceAnalyzer()
        self._run_task(
            lambda progress: analyzer.analyze(paths, auto_order=True, progress=progress),
            self._analysis_finished,
        )

    def _analysis_finished(self, result: object) -> None:
        analysis = result
        assert isinstance(analysis, AnalysisResult)
        self._analysis = analysis
        ordered = analysis.ordered_paths
        self._apply_order(ordered)
        self.create_button.setEnabled(True)
        if analysis.errors:
            message = "Automatic ordering needs attention: " + " ".join(analysis.errors)
        else:
            message = f"Circular sequence verified ({analysis.confidence:.0f}% confidence)."
        if analysis.warnings:
            message += " " + " ".join(analysis.warnings)
        self.status_label.setText(message)

    def _apply_order(self, paths: list[Path]) -> None:
        items: dict[Path, QListWidgetItem] = {}
        while self.photo_list.count():
            item = self.photo_list.takeItem(0)
            items[Path(item.data(Qt.ItemDataRole.UserRole)).resolve()] = item
        self._applying_order = True
        try:
            for path in paths:
                item = items[path.resolve()]
                self.photo_list.addItem(item)
            self.photo_list.renumber()
        finally:
            self._applying_order = False

    def _manual_order_changed(self) -> None:
        if self._applying_order or self._busy:
            return
        QTimer.singleShot(0, self.photo_list.renumber)
        self._analysis = None
        self.create_button.setEnabled(self.photo_list.count() >= 3)
        self.status_label.setText(
            "Manual order changed. Create Panorama will verify every neighboring overlap, including last-to-first."
        )

    def _start_stitching(self) -> None:
        paths = self.photo_list.paths()
        if len(paths) < 3 or self._busy:
            return
        stitcher = CylindricalStitcher()
        # The displayed order is authoritative now, whether it came from the
        # analyzer or from manual drag-and-drop.
        self._run_task(
            lambda progress: stitcher.create(paths, auto_order=False, progress=progress),
            self._stitching_finished,
        )

    def _stitching_finished(self, result: object) -> None:
        stitch_result = result
        assert isinstance(stitch_result, StitchResult)
        self._result = stitch_result
        self.viewer.set_rgb(stitch_result.image_rgb)
        self.result_info.setText(
            f"{stitch_result.output_width:,} × {stitch_result.output_height:,} px · cylindrical 360°"
        )
        self.save_button.setEnabled(True)
        message = "Panorama created successfully. Preview the wrap boundary, then save the full-resolution result."
        if stitch_result.warnings:
            message += " " + " ".join(stitch_result.warnings)
        self.status_label.setText(message)

    def _save_result(self) -> None:
        if self._result is None:
            return
        filename, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save 360 panorama",
            str(Path.home() / "panorama360.jpg"),
            "JPEG image (*.jpg *.jpeg);;PNG image (*.png);;TIFF image (*.tif *.tiff);;WebP image (*.webp)",
        )
        if not filename:
            return
        try:
            save_rgb(self._result.image_rgb, filename)
        except PanoramaError as exc:
            QMessageBox.critical(self, exc.title, exc.user_message())
            return
        self.status_label.setText(f"Panorama saved to {filename}")

    def _run_task(
        self,
        operation: Callable[[Callable[[int, str], None]], object],
        on_success: Callable[[object], None],
    ) -> None:
        self._set_busy(True)
        task = _Task(operation)
        self._active_task = task
        task.signals.progress.connect(self._progress_updated)
        task.signals.succeeded.connect(lambda result: self._task_succeeded(result, on_success))
        task.signals.failed.connect(self._task_failed)
        self._thread_pool.start(task)

    def _progress_updated(self, value: int, message: str) -> None:
        self.progress_bar.setValue(max(0, min(100, value)))
        self.progress_bar.setFormat(f"{message}  %p%")
        self.status_label.setText(message)

    def _task_succeeded(self, result: object, handler: Callable[[object], None]) -> None:
        self._set_busy(False)
        handler(result)
        self._active_task = None

    def _task_failed(self, title: str, message: str, technical: str) -> None:
        self._set_busy(False)
        self._active_task = None
        self.status_label.setText(message.replace("\n", " "))
        box = QMessageBox(QMessageBox.Icon.Warning, title, message, parent=self)
        if technical:
            box.setDetailedText(technical)
        box.exec()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.progress_bar.setVisible(busy)
        if busy:
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Starting…")
        self.photo_list.setEnabled(not busy)
        self.select_button.setEnabled(not busy)
        self.remove_button.setEnabled(not busy)
        enough = self.photo_list.count() >= 3
        self.analyze_button.setEnabled(not busy and enough)
        self.create_button.setEnabled(not busy and enough)
        self.save_button.setEnabled(not busy and self._result is not None)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #f3f4f6; color: #111827; }
            QLabel#title { font-size: 26px; font-weight: 700; }
            QLabel#subtitle { color: #4b5563; font-size: 13px; }
            QLabel#sectionTitle { font-size: 16px; font-weight: 700; }
            QLabel#hint { color: #6b7280; font-size: 12px; }
            QLabel#status { background: #eef2ff; color: #3730a3; border-radius: 7px;
                            padding: 8px 10px; font-size: 12px; }
            QFrame#panel { background: white; border: 1px solid #e5e7eb; border-radius: 10px; }
            QListWidget { background: #f9fafb; border: 1px dashed #cbd5e1; border-radius: 8px;
                          padding: 7px; outline: none; }
            QListWidget::item { background: white; border: 1px solid #e5e7eb; border-radius: 7px;
                                padding: 5px; color: #374151; }
            QListWidget::item:selected { border: 2px solid #4f46e5; background: #eef2ff; }
            QPushButton { min-height: 34px; padding: 0 14px; border-radius: 7px; font-weight: 600; }
            QPushButton#primaryButton { background: #4f46e5; color: white; border: none; }
            QPushButton#primaryButton:hover { background: #4338ca; }
            QPushButton#secondaryButton { background: white; color: #374151; border: 1px solid #d1d5db; }
            QPushButton#secondaryButton:hover { background: #f9fafb; border-color: #9ca3af; }
            QPushButton:disabled { background: #e5e7eb; color: #9ca3af; border-color: #e5e7eb; }
            QProgressBar { height: 18px; border: 1px solid #d1d5db; border-radius: 6px;
                           background: #f9fafb; text-align: center; }
            QProgressBar::chunk { background: #4f46e5; border-radius: 5px; }
            QSplitter::handle { height: 8px; background: transparent; }
            """
        )
