import sys
import os
import csv
import threading
import cv2
import numpy as np
import sqlite3
from datetime import datetime
from collections import defaultdict, deque

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox, QSlider, QAbstractItemView, QComboBox,
    QDoubleSpinBox, QMessageBox, QCheckBox, QGraphicsDropShadowEffect,
    QTabWidget, QTextEdit, QSplitter, QToolBar, QDateEdit, QSpinBox,
    QSizePolicy, QProgressDialog
)
from PyQt5.QtGui import QImage, QPixmap, QMouseEvent, QColor, QFont, QPainter, QPen
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QRect, QDate

from ultralytics import YOLO
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ============================================================
# 全局配色
# ============================================================
class Colors:
    PRIMARY = "#6366F1"
    PRIMARY_HOVER = "#818CF8"
    PRIMARY_DARK = "#4F46E5"
    BG_DARK = "#0F172A"
    BG_CARD = "#1E293B"
    BG_CARD_HOVER = "#334155"
    BG_INPUT = "#0F172A"
    TEXT_PRIMARY = "#F8FAFC"
    TEXT_SECONDARY = "#94A3B8"
    TEXT_MUTED = "#64748B"
    SUCCESS = "#10B981"
    WARNING = "#F59E0B"
    ERROR = "#EF4444"
    INFO = "#3B82F6"
    BORDER = "#334155"
    BORDER_LIGHT = "#475569"


# ============================================================
# 模型加载工作线程
# ============================================================
class ModelLoadWorker(QThread):
    finished = pyqtSignal(object, str)  # (model, model_name)
    error = pyqtSignal(str)

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def run(self):
        try:
            model = YOLO(self.model_name)
            self.finished.emit(model, self.model_name)
        except Exception as e:
            self.error.emit(str(e))


# 推理工作线程
# ============================================================
class Worker(QThread):
    frame_ready = pyqtSignal(np.ndarray, object, float)

    def __init__(self):
        super().__init__()
        self.model = None
        self.conf_thresh = 0.5
        self.is_tracking = False
        self._running = True
        self._busy = False
        self._frame = None
        self._lock = threading.Lock()
        self._event = threading.Event()

    @property
    def is_busy(self):
        return self._busy

    def submit_frame(self, frame):
        with self._lock:
            self._frame = frame
        self._event.set()

    def run(self):
        while self._running:
            self._event.wait(timeout=0.05)
            if not self._running:
                break

            frame = None
            with self._lock:
                if self._frame is not None:
                    frame = self._frame
                    self._frame = None
            self._event.clear()

            if frame is not None and self.model is not None:
                self._busy = True
                try:
                    t0 = cv2.getTickCount()
                    if self.is_tracking:
                        results = self.model.track(
                            source=frame, classes=[0],
                            conf=self.conf_thresh, verbose=False, persist=True
                        )
                    else:
                        results = self.model.predict(
                            source=frame, classes=[0],
                            conf=self.conf_thresh, verbose=False
                        )
                    dt = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000
                    self.frame_ready.emit(frame, results, dt)
                except Exception as e:
                    print(f"[Worker] 推理异常: {e}")
                self._busy = False

    def stop(self):
        self._running = False
        self._event.set()
        self.wait(3000)


# ============================================================
# 数据库管理
# ============================================================
class DatabaseManager:
    def __init__(self, db_path="detection_data.db"):
        self.db_path = db_path
        self._tracking_buffer = []
        self._buffer_lock = threading.Lock()
        self.init_database()

    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS detection_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL, source_type TEXT NOT NULL,
            source_path TEXT, model_name TEXT NOT NULL,
            confidence_threshold REAL, tracking_enabled INTEGER,
            total_persons INTEGER, fps REAL, processing_time REAL
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS alarm_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL, alarm_type TEXT NOT NULL,
            person_count INTEGER, zone_coordinates TEXT,
            snapshot_path TEXT, description TEXT
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS tracking_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL, track_id INTEGER,
            x_center REAL, y_center REAL,
            x1 REAL, y1 REAL, x2 REAL, y2 REAL,
            confidence REAL, behavior_type TEXT
        )''')
        conn.commit()
        conn.close()

    def insert_detection_record(self, record):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''INSERT INTO detection_records
            (timestamp,source_type,source_path,model_name,
             confidence_threshold,tracking_enabled,total_persons,fps,processing_time)
            VALUES (?,?,?,?,?,?,?,?,?)''', (
            record.get('timestamp', datetime.now().isoformat()),
            record.get('source_type', 'image'),
            record.get('source_path', ''),
            record.get('model_name', 'yolov8n.pt'),
            record.get('confidence_threshold', 0.5),
            record.get('tracking_enabled', 0),
            record.get('total_persons', 0),
            record.get('fps', 0.0),
            record.get('processing_time', 0.0),
        ))
        conn.commit()
        conn.close()

    def insert_alarm_event(self, event):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''INSERT INTO alarm_events
            (timestamp,alarm_type,person_count,zone_coordinates,snapshot_path,description)
            VALUES (?,?,?,?,?,?)''', (
            event.get('timestamp', datetime.now().isoformat()),
            event.get('alarm_type', 'zone_intrusion'),
            event.get('person_count', 0),
            event.get('zone_coordinates', ''),
            event.get('snapshot_path', ''),
            event.get('description', ''),
        ))
        conn.commit()
        conn.close()

    def buffer_tracking_data(self, data):
        with self._buffer_lock:
            self._tracking_buffer.append(data)

    def flush_tracking_buffer(self):
        with self._buffer_lock:
            if not self._tracking_buffer:
                return
            batch = list(self._tracking_buffer)
            self._tracking_buffer.clear()
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.executemany('''INSERT INTO tracking_data
            (timestamp,track_id,x_center,y_center,x1,y1,x2,y2,confidence,behavior_type)
            VALUES (?,?,?,?,?,?,?,?,?,?)''', [
            (d.get('timestamp', datetime.now().isoformat()),
             d.get('track_id', 0), d.get('x_center', 0.0), d.get('y_center', 0.0),
             d.get('x1', 0.0), d.get('y1', 0.0), d.get('x2', 0.0), d.get('y2', 0.0),
             d.get('confidence', 0.0), d.get('behavior_type', 'moving'))
            for d in batch
        ])
        conn.commit()
        conn.close()

    def get_detection_records(self, start_date=None, end_date=None):
        conn = sqlite3.connect(self.db_path)
        q = "SELECT * FROM detection_records WHERE 1=1"
        params = []
        if start_date:
            q += " AND timestamp >= ?"; params.append(start_date)
        if end_date:
            q += " AND timestamp <= ?"; params.append(end_date + "T23:59:59")
        q += " ORDER BY timestamp DESC"
        rows = conn.execute(q, params).fetchall()
        conn.close()
        return rows

    def get_alarm_events(self, start_date=None, end_date=None):
        conn = sqlite3.connect(self.db_path)
        q = "SELECT * FROM alarm_events WHERE 1=1"
        params = []
        if start_date:
            q += " AND timestamp >= ?"; params.append(start_date)
        if end_date:
            q += " AND timestamp <= ?"; params.append(end_date + "T23:59:59")
        q += " ORDER BY timestamp DESC"
        rows = conn.execute(q, params).fetchall()
        conn.close()
        return rows

    def get_statistics(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM detection_records")
        total_det = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM alarm_events")
        total_alm = c.fetchone()[0]
        c.execute("SELECT AVG(total_persons) FROM detection_records")
        avg_p = c.fetchone()[0] or 0
        c.execute("SELECT AVG(fps) FROM detection_records")
        avg_f = c.fetchone()[0] or 0
        conn.close()
        return {
            'total_detections': total_det,
            'total_alarms': total_alm,
            'avg_persons': round(avg_p, 2),
            'avg_fps': round(avg_f, 2),
        }

    def export_to_csv(self, table_name, output_path):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(f"SELECT * FROM {table_name}")
        rows = c.fetchall()
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow([d[0] for d in c.description])
            w.writerows(rows)
        conn.close()

    def export_to_excel(self, output_path):
        if not HAS_PANDAS:
            return False
        conn = sqlite3.connect(self.db_path)
        dfs = {}
        for t in ('detection_records', 'alarm_events', 'tracking_data'):
            try:
                dfs[t] = pd.read_sql_query(f"SELECT * FROM {t}", conn)
            except Exception:
                pass
        conn.close()
        if dfs:
            with pd.ExcelWriter(output_path, engine='openpyxl') as wr:
                for name, df in dfs.items():
                    df.to_excel(wr, sheet_name=name, index=False)
            return True
        return False


# ============================================================
# 热力图生成器（向量化）
# ============================================================
class HeatmapGenerator:
    def __init__(self, width=640, height=480, grid_size=20):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.heatmap = np.zeros((height, width), dtype=np.float32)

    def update_position(self, x, y, weight=1.0):
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        radius = self.grid_size * 2
        y0 = max(0, y - radius)
        y1 = min(self.height, y + radius)
        x0 = max(0, x - radius)
        x1 = min(self.width, x + radius)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist = np.sqrt((xx - x) ** 2.0 + (yy - y) ** 2.0)
        mask = dist < radius
        self.heatmap[y0:y1, x0:x1] += weight * (1.0 - dist / radius) * mask

    def get_heatmap_image(self):
        normed = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.applyColorMap(normed.astype(np.uint8), cv2.COLORMAP_JET)

    def reset(self):
        self.heatmap = np.zeros((self.height, self.width), dtype=np.float32)


# ============================================================
# 轨迹管理
# ============================================================
class TrajectoryManager:
    def __init__(self, max_history=100):
        self.trajectories = defaultdict(lambda: deque(maxlen=max_history))

    def update(self, track_id, x, y):
        self.trajectories[track_id].append((x, y))

    def get_trajectory(self, tid):
        return list(self.trajectories[tid])

    def get_all_trajectories(self):
        return dict(self.trajectories)

    def draw_trajectories(self, image, thickness=2):
        palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
        ]
        for tid, pts in self.trajectories.items():
            if len(pts) < 2:
                continue
            color = palette[tid % len(palette)]
            for i in range(1, len(pts)):
                cv2.line(image,
                         (int(pts[i - 1][0]), int(pts[i - 1][1])),
                         (int(pts[i][0]), int(pts[i][1])),
                         color, thickness)
            cv2.circle(image, (int(pts[-1][0]), int(pts[-1][1])), 5, color, -1)

    def clear(self):
        self.trajectories.clear()


# ============================================================
# 行为分析（阈值归一化）
# ============================================================
class BehaviorAnalyzer:
    def __init__(self, history_len=60):
        self.position_history = defaultdict(lambda: deque(maxlen=history_len))

    def analyze(self, track_id, x, y, frame_width, frame_height):
        self.position_history[track_id].append((x, y))
        if len(self.position_history[track_id]) < 10:
            return 'unknown'
        positions = list(self.position_history[track_id])
        recent = positions[-10:]
        diagonal = np.sqrt(frame_width ** 2 + frame_height ** 2)
        norm = diagonal / 1000.0
        movements = []
        for i in range(1, len(recent)):
            d = np.sqrt((recent[i][0] - recent[i - 1][0]) ** 2 +
                        (recent[i][1] - recent[i - 1][1]) ** 2)
            movements.append(d / norm)
        avg = np.mean(movements)
        if avg < 2.0:
            return 'stationary'
        elif avg < 5.0:
            return 'walking'
        else:
            return 'running'

    def detect_crowding(self, positions, threshold=5, radius=100):
        if len(positions) < threshold:
            return False, []
        crowded = []
        for i, p1 in enumerate(positions):
            group = [p1]
            for j, p2 in enumerate(positions):
                if i != j:
                    if np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) < radius:
                        group.append(p2)
            if len(group) >= threshold:
                crowded.append(group)
        return len(crowded) > 0, crowded

    def clear(self):
        self.position_history.clear()


# ============================================================
# UI 组件
# ============================================================
class CardWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QWidget {{ background-color: {Colors.BG_CARD};
                       border-radius: 12px; border: 1px solid {Colors.BORDER}; }}
        """)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20); shadow.setXOffset(0); shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 60))
        self.setGraphicsEffect(shadow)


class ModernButton(QPushButton):
    def __init__(self, text, primary=False):
        super().__init__(text)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(42)
        self.setFont(QFont("Segoe UI", 10, QFont.Medium))
        if primary:
            self.setStyleSheet(f"""
                QPushButton {{ background-color: {Colors.PRIMARY}; color: {Colors.TEXT_PRIMARY};
                    border: none; border-radius: 8px; padding: 8px 16px; font-weight: 500; }}
                QPushButton:hover {{ background-color: {Colors.PRIMARY_HOVER}; }}
                QPushButton:pressed {{ background-color: {Colors.PRIMARY_DARK}; }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{ background-color: {Colors.BG_CARD}; color: {Colors.TEXT_PRIMARY};
                    border: 1px solid {Colors.BORDER}; border-radius: 8px; padding: 8px 16px; font-weight: 500; }}
                QPushButton:hover {{ background-color: {Colors.BG_CARD_HOVER}; border-color: {Colors.BORDER_LIGHT}; }}
                QPushButton:pressed {{ background-color: {Colors.BG_INPUT}; }}
            """)


class StatusIndicator(QWidget):
    def __init__(self, text, color, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(36)
        lay = QHBoxLayout(self); lay.setContentsMargins(12, 8, 12, 8); lay.setSpacing(10)
        self.dot = QLabel(); self.dot.setFixedSize(10, 10)
        self.dot.setStyleSheet(f"background-color: {color}; border-radius: 5px;")
        self.label = QLabel(text)
        self.label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 13px; font-weight: 500;")
        lay.addWidget(self.dot); lay.addWidget(self.label); lay.addStretch()

    def update_status(self, text, color):
        self.dot.setStyleSheet(f"background-color: {color}; border-radius: 5px;")
        self.label.setText(text)


# ============================================================
# Matplotlib 嵌入组件
# ============================================================
class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.addWidget(self.canvas)
        self.figure.patch.set_facecolor(Colors.BG_INPUT)
        self.canvas.setStyleSheet(f"background-color: {Colors.BG_INPUT};")

    def plot_line_chart(self, x, y, title="", xlabel="", ylabel="", color=Colors.PRIMARY):
        self.figure.clear(); ax = self.figure.add_subplot(111)
        ax.plot(x, y, color=color, linewidth=2, marker='o', markersize=4)
        ax.set_title(title, color=Colors.TEXT_PRIMARY, fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel, color=Colors.TEXT_SECONDARY)
        ax.set_ylabel(ylabel, color=Colors.TEXT_SECONDARY)
        ax.tick_params(colors=Colors.TEXT_SECONDARY)
        for spine in ax.spines.values(): spine.set_color(Colors.BORDER)
        ax.set_facecolor(Colors.BG_INPUT)
        self.figure.tight_layout(); self.canvas.draw()

    def plot_pie_chart(self, labels, sizes, title=""):
        self.figure.clear(); ax = self.figure.add_subplot(111)
        palette = [Colors.PRIMARY, Colors.SUCCESS, Colors.WARNING, Colors.ERROR, Colors.INFO]
        wedges, texts, autos = ax.pie(
            sizes, labels=labels, autopct='%1.1f%%',
            colors=palette[:len(labels)], explode=[0.05] * len(sizes), startangle=90)
        for t in texts: t.set_color(Colors.TEXT_PRIMARY)
        for a in autos: a.set_color(Colors.TEXT_PRIMARY); a.set_fontweight('bold')
        ax.set_title(title, color=Colors.TEXT_PRIMARY, fontsize=12, fontweight='bold')
        self.figure.tight_layout(); self.canvas.draw()


class HeatmapDisplayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.addWidget(self.canvas)
        self.figure.patch.set_facecolor(Colors.BG_INPUT)
        self.canvas.setStyleSheet(f"background-color: {Colors.BG_INPUT};")

    def update_heatmap(self, heatmap_data):
        self.figure.clear(); ax = self.figure.add_subplot(111)
        if heatmap_data is None or np.sum(heatmap_data) == 0:
            ax.text(0.5, 0.5, '暂无热力图数据\n请开启追踪并播放视频', ha='center', va='center',
                    fontsize=14, color=Colors.TEXT_SECONDARY)
            ax.set_title('行人活动热力图', color=Colors.TEXT_PRIMARY, fontsize=12, fontweight='bold')
            ax.axis('off')
        else:
            normed = cv2.normalize(heatmap_data, None, 0, 255, cv2.NORM_MINMAX)
            colored = cv2.applyColorMap(normed.astype(np.uint8), cv2.COLORMAP_JET)
            rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            ax.imshow(rgb)
            ax.set_title('行人活动热力图', color=Colors.TEXT_PRIMARY, fontsize=12, fontweight='bold')
            ax.axis('off')
        self.figure.tight_layout(); self.canvas.draw()


class TrajectoryAnalysisWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.addWidget(self.canvas)
        self.figure.patch.set_facecolor(Colors.BG_INPUT)
        self.canvas.setStyleSheet(f"background-color: {Colors.BG_INPUT};")

    def plot_trajectories(self, trajectories):
        self.figure.clear(); ax = self.figure.add_subplot(111)
        if not trajectories:
            ax.text(0.5, 0.5, '暂无轨迹数据\n请开启追踪并播放视频', ha='center', va='center',
                    fontsize=14, color=Colors.TEXT_SECONDARY)
            ax.set_title('行人运动轨迹', color=Colors.TEXT_PRIMARY, fontsize=12, fontweight='bold')
            ax.axis('off')
        else:
            palette = ['#6366F1', '#10B981', '#F59E0B', '#EF4444', '#3B82F6', '#8B5CF6', '#EC4899', '#14B8A6']
            first = True
            for tid, pts in trajectories.items():
                if len(pts) < 2: continue
                c = palette[tid % len(palette)]
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                ax.plot(xs, ys, color=c, linewidth=2, alpha=0.7, label=f'ID_{tid}')
                ax.plot(xs[0], ys[0], 'go', markersize=8,
                        label='起点' if first else "")
                ax.plot(xs[-1], ys[-1], 'ro', markersize=8,
                        label='终点' if first else "")
                first = False
            ax.set_title('行人运动轨迹', color=Colors.TEXT_PRIMARY, fontsize=12, fontweight='bold')
            ax.set_xlabel('X 坐标', color=Colors.TEXT_SECONDARY)
            ax.set_ylabel('Y 坐标', color=Colors.TEXT_SECONDARY)
            ax.tick_params(colors=Colors.TEXT_SECONDARY)
            for spine in ax.spines.values(): spine.set_color(Colors.BORDER)
            ax.set_facecolor(Colors.BG_INPUT)
            ax.legend(loc='upper right', fontsize=8)
        self.figure.tight_layout(); self.canvas.draw()


# ============================================================
# 主窗口
# ============================================================
class YoloVideoPro(QMainWindow):
    def __init__(self):
        super().__init__()

        # 模型
        self.current_model_name = 'yolov8n.pt'
        self.model = YOLO(self.current_model_name)

        # 视频
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_timer_tick)
        self.is_paused = False
        self.total_frames = 0
        self.current_res_img = None
        self.original_pixmap = None
        self._base_pixmap = None
        self._last_frame = None  # 【修复】缓存上一帧，Worker忙时可复用

        # ROI 绘制
        self.is_drawing = False
        self.roi_start_point = None
        self.roi_end_point = None
        self.alarm_zone = None

        # 追踪
        self.is_tracking_enabled = False

        # 推理工作线程
        self.worker = Worker()
        self.worker.model = self.model  # 【关键修复】初始化时就绑定模型
        self.worker.frame_ready.connect(self.on_worker_result)
        self.worker.start()

        # 模块
        self.db = DatabaseManager()
        self.heatmap = HeatmapGenerator()
        self.trajectory = TrajectoryManager()
        self.behavior = BehaviorAnalyzer()
        self._model_loader = None

        # 统计
        self.person_count_history = []
        self.time_history = []
        self.frame_count = 0
        self._db_flush_counter = 0
        self._alarm_was_active = False

        self.initUI()

        # 初始渲染热力图占位文字
        self.heatmap_widget.update_heatmap(None)

        # 加载历史报警记录
        self._load_alarm_history()

    # ----------------------------------------------------------
    # UI 初始化
    # ----------------------------------------------------------
    def initUI(self):
        self.setWindowTitle('YOLO 图像行人检测系统')
        self.resize(1600, 900)
        self.setMinimumSize(1400, 800)
        self.setStyleSheet(self._global_stylesheet())

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ---- 工具栏 ----
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setFixedHeight(60)
        toolbar.setStyleSheet(f"""
            QToolBar {{ background-color: {Colors.BG_CARD};
                       border-bottom: 1px solid {Colors.BORDER}; padding: 8px; spacing: 8px; }}
        """)

        self.btn_img = ModernButton('📷 打开图片', primary=True)
        self.btn_img.setFixedWidth(120)
        self.btn_video = ModernButton('🎬 打开视频')
        self.btn_video.setFixedWidth(120)
        self.btn_camera = ModernButton('📹 摄像头')
        self.btn_camera.setFixedWidth(100)
        self.btn_play_pause = ModernButton('▶ 播放')
        self.btn_play_pause.setFixedWidth(90)
        self.btn_stop = ModernButton('⏹ 停止')
        self.btn_stop.setFixedWidth(80)

        toolbar.addWidget(self.btn_img)
        toolbar.addWidget(self.btn_video)
        toolbar.addWidget(self.btn_camera)
        toolbar.addSeparator()
        toolbar.addWidget(self.btn_play_pause)
        toolbar.addWidget(self.btn_stop)

        spacer = QWidget(); spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        toolbar.addWidget(spacer)

        self.check_track = QCheckBox("启用追踪")
        self.check_track.setStyleSheet(f"""
            QCheckBox {{ color: {Colors.TEXT_PRIMARY}; font-size: 13px; font-weight: 500; spacing: 8px; }}
            QCheckBox::indicator {{ width: 18px; height: 18px; border-radius: 4px;
                border: 2px solid {Colors.BORDER}; background-color: {Colors.BG_INPUT}; }}
            QCheckBox::indicator:checked {{ background-color: {Colors.PRIMARY}; border-color: {Colors.PRIMARY}; }}
        """)
        toolbar.addWidget(self.check_track)

        self.btn_clear_zone = ModernButton('🗑 清除警戒')
        self.btn_clear_zone.setFixedWidth(100)
        self.btn_clear_zone.clicked.connect(self.clear_alarm_zone)
        toolbar.addWidget(self.btn_clear_zone)

        self.btn_export = ModernButton('💾 导出数据')
        self.btn_export.setFixedWidth(100)
        toolbar.addWidget(self.btn_export)

        self.btn_export_video = ModernButton('🎥 导出视频')
        self.btn_export_video.setFixedWidth(110)
        toolbar.addWidget(self.btn_export_video)

        main_layout.addWidget(toolbar)

        # ---- 内容区 ----
        content_splitter = QSplitter(Qt.Horizontal)

        # 左侧：画面
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(16, 16, 8, 16)
        left_layout.setSpacing(12)

        display_card = CardWidget()
        display_layout = QVBoxLayout(display_card)
        display_layout.setContentsMargins(16, 16, 16, 16)

        self.label_display = QLabel('📷 请打开图片或视频开始检测')
        self.label_display.setAlignment(Qt.AlignCenter)
        self.label_display.setStyleSheet(f"""
            background-color: {Colors.BG_INPUT}; border: 2px dashed {Colors.BORDER};
            border-radius: 8px; font-size: 18px; color: {Colors.TEXT_MUTED};
        """)
        self.label_display.setScaledContents(False)
        self.label_display.setMinimumSize(640, 480)
        self.label_display.setMouseTracking(True)
        self.label_display.mousePressEvent = self.mouse_press
        self.label_display.mouseMoveEvent = self.mouse_move
        self.label_display.mouseReleaseEvent = self.mouse_release

        display_layout.addWidget(self.label_display)
        left_layout.addWidget(display_card, stretch=1)

        progress_card = CardWidget()
        progress_layout = QVBoxLayout(progress_card)
        progress_layout.setContentsMargins(16, 12, 16, 12)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderMoved.connect(self._on_slider_moved)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self._slider_dragging = False
        progress_layout.addWidget(self.slider)
        left_layout.addWidget(progress_card)

        left_widget.setLayout(left_layout)

        # 右侧：选项卡
        right_tabs = QTabWidget()
        right_tabs.setFixedWidth(420)

        # Tab 0: 检测面板
        detection_panel = QWidget()
        det_lay = QVBoxLayout(detection_panel)
        det_lay.setContentsMargins(12, 12, 12, 12)
        det_lay.setSpacing(12)

        settings_group = QGroupBox("⚙ 参数设置")
        sv = QVBoxLayout(settings_group); sv.setSpacing(12)
        ml = QVBoxLayout()
        ml.addWidget(self._secondary_label("模型选择"))
        self.combo_model = QComboBox()
        self.combo_model.addItems(["yolov8n.pt (轻量)", "yolov5nu.pt", "yolov8s.pt"])
        self.combo_model.setMinimumHeight(36)
        ml.addWidget(self.combo_model); sv.addLayout(ml)

        cl = QVBoxLayout()
        cl.addWidget(self._secondary_label("置信度阈值"))
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.1, 1.0); self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.5); self.spin_conf.setMinimumHeight(36)
        cl.addWidget(self.spin_conf); sv.addLayout(cl)
        det_lay.addWidget(settings_group)

        stats_group = QGroupBox("📊 实时统计")
        stv = QVBoxLayout(stats_group); stv.setSpacing(12)
        self.status_indicator = StatusIndicator("系统就绪", Colors.SUCCESS)
        stv.addWidget(self.status_indicator)
        self.lbl_count = QLabel("🚶 行人总数: 0")
        self.lbl_count.setStyleSheet(f"font-size: 16px; font-weight: 600; color: {Colors.SUCCESS};")
        self.lbl_speed = QLabel("⏱ 推理耗时: 0.0 ms")
        self.lbl_speed.setStyleSheet(f"font-size: 14px; color: {Colors.WARNING};")
        self.lbl_fps = QLabel("📈 FPS: 0.0")
        self.lbl_fps.setStyleSheet(f"font-size: 14px; color: {Colors.INFO};")
        stv.addWidget(self.lbl_count)
        stv.addWidget(self.lbl_speed)
        stv.addWidget(self.lbl_fps)
        det_lay.addWidget(stats_group)

        table_group = QGroupBox("📋 检测明细")
        tl = QVBoxLayout(table_group); tl.setContentsMargins(8, 8, 8, 8)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['ID', '类别', '置信度', '行为'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        tl.addWidget(self.table)
        det_lay.addWidget(table_group)
        right_tabs.addTab(detection_panel, "🔍 检测")

        # Tab 1: 数据分析
        analysis_panel = QWidget()
        ana_lay = QVBoxLayout(analysis_panel)
        ana_lay.setContentsMargins(12, 12, 12, 12); ana_lay.setSpacing(12)
        cg = QGroupBox("📈 行人数量变化")
        cl2 = QVBoxLayout(cg); cl2.setContentsMargins(8, 8, 8, 8)
        self.chart_widget = MatplotlibWidget()
        cl2.addWidget(self.chart_widget)
        ana_lay.addWidget(cg)

        sg = QGroupBox("📊 统计汇总")
        sl2 = QVBoxLayout(sg); sl2.setSpacing(8)
        self.lbl_total_detections = QLabel("总检测次数: 0")
        self.lbl_total_detections.setStyleSheet(f"font-size: 14px; color: {Colors.TEXT_PRIMARY};")
        self.lbl_total_alarms = QLabel("总报警次数: 0")
        self.lbl_total_alarms.setStyleSheet(f"font-size: 14px; color: {Colors.TEXT_PRIMARY};")
        self.lbl_avg_persons = QLabel("平均行人数量: 0")
        self.lbl_avg_persons.setStyleSheet(f"font-size: 14px; color: {Colors.TEXT_PRIMARY};")
        self.lbl_avg_fps = QLabel("平均 FPS: 0")
        self.lbl_avg_fps.setStyleSheet(f"font-size: 14px; color: {Colors.TEXT_PRIMARY};")
        sl2.addWidget(self.lbl_total_detections)
        sl2.addWidget(self.lbl_total_alarms)
        sl2.addWidget(self.lbl_avg_persons)
        sl2.addWidget(self.lbl_avg_fps)
        self.btn_refresh_stats = ModernButton("🔄 刷新统计")
        self.btn_refresh_stats.clicked.connect(self.refresh_statistics)
        sl2.addWidget(self.btn_refresh_stats)
        ana_lay.addWidget(sg)
        right_tabs.addTab(analysis_panel, "📊 分析")

        # Tab 2: 热力图
        heatmap_panel = QWidget()
        hm_lay = QVBoxLayout(heatmap_panel)
        hm_lay.setContentsMargins(12, 12, 12, 12); hm_lay.setSpacing(12)
        hg = QGroupBox("🔥 行人活动热力图")
        hml = QVBoxLayout(hg); hml.setContentsMargins(8, 8, 8, 8)
        self.heatmap_widget = HeatmapDisplayWidget()
        hml.addWidget(self.heatmap_widget)
        self.btn_clear_heatmap = ModernButton("🗑 清除热力图")
        self.btn_clear_heatmap.clicked.connect(self.clear_heatmap)
        hml.addWidget(self.btn_clear_heatmap)
        hm_lay.addWidget(hg)
        right_tabs.addTab(heatmap_panel, "🔥 热力图")

        # Tab 3: 轨迹
        trajectory_panel = QWidget()
        tr_lay = QVBoxLayout(trajectory_panel)
        tr_lay.setContentsMargins(12, 12, 12, 12); tr_lay.setSpacing(12)
        trg = QGroupBox("📍 行人运动轨迹")
        trl = QVBoxLayout(trg); trl.setContentsMargins(8, 8, 8, 8)
        self.trajectory_widget = TrajectoryAnalysisWidget()
        trl.addWidget(self.trajectory_widget)
        tsh = QHBoxLayout()
        self.lbl_total_tracks = QLabel("总轨迹数: 0")
        self.lbl_total_tracks.setStyleSheet(f"font-size: 14px; color: {Colors.TEXT_PRIMARY};")
        self.lbl_avg_distance = QLabel("平均移动距离: 0")
        self.lbl_avg_distance.setStyleSheet(f"font-size: 14px; color: {Colors.TEXT_PRIMARY};")
        tsh.addWidget(self.lbl_total_tracks); tsh.addWidget(self.lbl_avg_distance)
        trl.addLayout(tsh)
        self.btn_update_trajectory = ModernButton("🔄 更新轨迹分析")
        self.btn_update_trajectory.clicked.connect(self.update_trajectory_analysis)
        trl.addWidget(self.btn_update_trajectory)
        tr_lay.addWidget(trg)
        right_tabs.addTab(trajectory_panel, "📍 轨迹")

        # Tab 4: 历史记录
        history_panel = QWidget()
        hist_lay = QVBoxLayout(history_panel)
        hist_lay.setContentsMargins(12, 12, 12, 12); hist_lay.setSpacing(12)
        fl = QHBoxLayout()
        fl.addWidget(QLabel("开始:"))
        self.date_start = QDateEdit(); self.date_start.setCalendarPopup(True)
        self.date_start.setDate(QDate.currentDate().addDays(-7)); self.date_start.setMinimumHeight(32)
        fl.addWidget(self.date_start)
        fl.addWidget(QLabel("结束:"))
        self.date_end = QDateEdit(); self.date_end.setCalendarPopup(True)
        self.date_end.setDate(QDate.currentDate()); self.date_end.setMinimumHeight(32)
        fl.addWidget(self.date_end)
        self.btn_query = ModernButton("🔍 查询")
        self.btn_query.setFixedWidth(80); self.btn_query.clicked.connect(self.query_history)
        fl.addWidget(self.btn_query)
        hist_lay.addLayout(fl)

        self.history_table = QTableWidget(0, 6)
        self.history_table.setHorizontalHeaderLabels(['时间', '类型', '模型', '行人', 'FPS', '耗时'])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.history_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.history_table.verticalHeader().setVisible(False)
        hist_lay.addWidget(self.history_table)
        right_tabs.addTab(history_panel, "📜 历史")

        # Tab 5: 报警
        alarm_panel = QWidget()
        alm_lay = QVBoxLayout(alarm_panel)
        alm_lay.setContentsMargins(12, 12, 12, 12); alm_lay.setSpacing(12)
        self.alarm_log = QTextEdit(); self.alarm_log.setReadOnly(True); self.alarm_log.setMinimumHeight(300)
        alm_lay.addWidget(self.alarm_log)
        self.btn_clear_alarm = ModernButton("🗑 清除报警日志")
        self.btn_clear_alarm.clicked.connect(self.clear_alarm_log)
        alm_lay.addWidget(self.btn_clear_alarm)
        right_tabs.addTab(alarm_panel, "🚨 报警")

        # Tab 6: 行为分析
        behavior_panel = QWidget()
        beh_lay = QVBoxLayout(behavior_panel)
        beh_lay.setContentsMargins(12, 12, 12, 12); beh_lay.setSpacing(12)
        bcg = QGroupBox("📊 行为分布图")
        bcl = QVBoxLayout(bcg); bcl.setContentsMargins(8, 8, 8, 8)
        self.behavior_chart_widget = MatplotlibWidget()
        bcl.addWidget(self.behavior_chart_widget)
        beh_lay.addWidget(bcg)
        bsg = QGroupBox("📋 行为统计")
        bsl = QVBoxLayout(bsg); bsl.setSpacing(8)
        self.lbl_stationary = QLabel("🟤 静止: 0")
        self.lbl_stationary.setStyleSheet(f"font-size: 14px; color: {Colors.TEXT_PRIMARY};")
        self.lbl_walking = QLabel("🟡 行走: 0")
        self.lbl_walking.setStyleSheet(f"font-size: 14px; color: {Colors.TEXT_PRIMARY};")
        self.lbl_running = QLabel("🔴 奔跑: 0")
        self.lbl_running.setStyleSheet(f"font-size: 14px; color: {Colors.TEXT_PRIMARY};")
        bsl.addWidget(self.lbl_stationary)
        bsl.addWidget(self.lbl_walking)
        bsl.addWidget(self.lbl_running)
        self.btn_update_behavior = ModernButton("🔄 更新行为分析")
        self.btn_update_behavior.clicked.connect(self.update_behavior_analysis)
        bsl.addWidget(self.btn_update_behavior)
        beh_lay.addWidget(bsg)
        right_tabs.addTab(behavior_panel, "🚶 行为")

        # Tab 7: 仪表盘
        dashboard_panel = QWidget()
        dash_lay = QVBoxLayout(dashboard_panel)
        dash_lay.setContentsMargins(12, 12, 12, 12); dash_lay.setSpacing(12)
        og = QGroupBox("📊 总览")
        ol = QVBoxLayout(og); ol.setSpacing(12)
        self.lbl_dashboard_total = QLabel("🔵 总检测次数: 0")
        self.lbl_dashboard_total.setStyleSheet(f"font-size: 16px; font-weight: 600; color: {Colors.PRIMARY};")
        self.lbl_dashboard_alarms = QLabel("🔴 总报警次数: 0")
        self.lbl_dashboard_alarms.setStyleSheet(f"font-size: 16px; font-weight: 600; color: {Colors.ERROR};")
        self.lbl_dashboard_avg_persons = QLabel("🟢 平均行人数量: 0")
        self.lbl_dashboard_avg_persons.setStyleSheet(f"font-size: 16px; font-weight: 600; color: {Colors.SUCCESS};")
        self.lbl_dashboard_avg_fps = QLabel("🟡 平均 FPS: 0")
        self.lbl_dashboard_avg_fps.setStyleSheet(f"font-size: 16px; font-weight: 600; color: {Colors.WARNING};")
        self.lbl_dashboard_tracks = QLabel("📍 总轨迹数: 0")
        self.lbl_dashboard_tracks.setStyleSheet(f"font-size: 16px; font-weight: 600; color: {Colors.INFO};")
        self.lbl_dashboard_crowding = QLabel("⚠ 聚集检测: 0")
        self.lbl_dashboard_crowding.setStyleSheet(f"font-size: 16px; font-weight: 600; color: {Colors.ERROR};")
        for w in (self.lbl_dashboard_total, self.lbl_dashboard_alarms,
                  self.lbl_dashboard_avg_persons, self.lbl_dashboard_avg_fps,
                  self.lbl_dashboard_tracks, self.lbl_dashboard_crowding):
            ol.addWidget(w)
        dash_lay.addWidget(og)

        dcg = QGroupBox("📈 趋势图")
        dcl = QVBoxLayout(dcg); dcl.setContentsMargins(8, 8, 8, 8)
        self.dashboard_chart_widget = MatplotlibWidget()
        dcl.addWidget(self.dashboard_chart_widget)
        dash_lay.addWidget(dcg)

        self.btn_refresh_dashboard = ModernButton("🔄 刷新仪表盘")
        self.btn_refresh_dashboard.clicked.connect(self.refresh_dashboard)
        dash_lay.addWidget(self.btn_refresh_dashboard)
        right_tabs.addTab(dashboard_panel, "📊 仪表盘")

        content_splitter.addWidget(left_widget)
        content_splitter.addWidget(right_tabs)
        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 1)

        main_layout.addWidget(content_splitter)
        container = QWidget(); container.setLayout(main_layout)
        self.setCentralWidget(container)
        self.statusBar().showMessage("就绪")

        # 信号连接
        self.btn_img.clicked.connect(self.detect_image)
        self.btn_video.clicked.connect(self.open_video)
        self.btn_camera.clicked.connect(self.open_camera)
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        self.btn_stop.clicked.connect(self.stop_all)
        self.btn_export.clicked.connect(self.export_data)
        self.btn_export_video.clicked.connect(self.export_video)
        self.combo_model.currentTextChanged.connect(self.change_model)

    # ----------------------------------------------------------
    # 样式表
    # ----------------------------------------------------------
    def _global_stylesheet(self):
        return f"""
            QMainWindow {{ background-color: {Colors.BG_DARK}; }}
            QLabel {{ color: {Colors.TEXT_PRIMARY}; font-family: "Segoe UI", "Microsoft YaHei"; }}
            QComboBox {{
                background-color: {Colors.BG_INPUT}; color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER}; border-radius: 6px;
                padding: 6px 12px; min-height: 32px;
            }}
            QComboBox:hover {{ border-color: {Colors.PRIMARY}; }}
            QComboBox::drop-down {{ border: none; padding-right: 8px; }}
            QDoubleSpinBox, QSpinBox {{
                background-color: {Colors.BG_INPUT}; color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER}; border-radius: 6px;
                padding: 6px 12px; min-height: 32px;
            }}
            QDoubleSpinBox:hover, QSpinBox:hover {{ border-color: {Colors.PRIMARY}; }}
            QSlider::groove:horizontal {{ height: 6px; background: {Colors.BORDER}; border-radius: 3px; }}
            QSlider::handle:horizontal {{
                background: {Colors.PRIMARY}; width: 18px; margin: -6px 0; border-radius: 9px;
            }}
            QSlider::handle:horizontal:hover {{ background: {Colors.PRIMARY_HOVER}; }}
            QGroupBox {{
                background-color: {Colors.BG_CARD}; border: 1px solid {Colors.BORDER};
                border-radius: 12px; margin-top: 12px; padding-top: 16px;
                font-weight: 600; font-size: 14px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin; left: 16px; padding: 0 8px; color: {Colors.TEXT_PRIMARY};
            }}
            QTabWidget::pane {{
                border: 1px solid {Colors.BORDER}; border-radius: 8px; background-color: {Colors.BG_CARD};
            }}
            QTabBar::tab {{
                background-color: {Colors.BG_INPUT}; color: {Colors.TEXT_SECONDARY};
                padding: 8px 16px; margin-right: 2px;
                border-top-left-radius: 6px; border-top-right-radius: 6px;
            }}
            QTabBar::tab:selected {{ background-color: {Colors.PRIMARY}; color: {Colors.TEXT_PRIMARY}; }}
            QTabBar::tab:hover {{ background-color: {Colors.BG_CARD_HOVER}; }}
            QTableWidget {{
                background-color: {Colors.BG_INPUT}; border: none; border-radius: 8px;
                gridline-color: {Colors.BORDER}; color: {Colors.TEXT_PRIMARY};
                alternate-background-color: #1a2332;
            }}
            QHeaderView::section {{
                background-color: {Colors.BG_CARD}; padding: 8px; border: none;
                border-bottom: 2px solid {Colors.BORDER}; font-weight: 600; color: {Colors.TEXT_SECONDARY};
            }}
            QTableWidget::item {{ padding: 8px; border: none; }}
            QTableWidget::item:selected {{ background-color: {Colors.PRIMARY_DARK}; }}
            QTextEdit {{
                background-color: {Colors.BG_INPUT}; color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER}; border-radius: 6px; padding: 8px;
            }}
            QMessageBox {{
                background-color: {Colors.BG_CARD};
            }}
            QMessageBox QLabel {{
                color: {Colors.TEXT_PRIMARY};
                font-size: 14px;
                min-width: 300px;
            }}
            QMessageBox QPushButton {{
                background-color: {Colors.PRIMARY};
                color: {Colors.TEXT_PRIMARY};
                border: none;
                border-radius: 6px;
                padding: 6px 20px;
                min-width: 60px;
                font-weight: 500;
            }}
            QMessageBox QPushButton:hover {{
                background-color: {Colors.PRIMARY_HOVER};
            }}
            QDateEdit {{
                background-color: {Colors.BG_INPUT}; color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER}; border-radius: 6px;
                padding: 6px 12px; min-height: 28px;
            }}
            QDateEdit:hover {{ border-color: {Colors.PRIMARY}; }}
            QDateEdit::drop-down {{ border: none; padding-right: 8px; }}
            QCalendarWidget {{
                background-color: {Colors.BG_CARD};
                color: {Colors.TEXT_PRIMARY};
            }}
            QCalendarWidget QWidget {{
                alternate-background-color: {Colors.BG_CARD};
            }}
            QCalendarWidget QAbstractItemView {{
                background-color: {Colors.BG_INPUT};
                color: {Colors.TEXT_PRIMARY};
                selection-background-color: {Colors.PRIMARY};
                selection-color: {Colors.TEXT_PRIMARY};
            }}
            QCalendarWidget QToolButton {{
                color: {Colors.TEXT_PRIMARY};
                background-color: {Colors.BG_CARD};
                border: none;
                padding: 4px;
            }}
            QCalendarWidget QMenu {{
                background-color: {Colors.BG_CARD};
                color: {Colors.TEXT_PRIMARY};
            }}
            QCalendarWidget QSpinBox {{
                background-color: {Colors.BG_INPUT};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
            }}
            QProgressDialog {{
                background-color: {Colors.BG_CARD};
                color: {Colors.TEXT_PRIMARY};
            }}
            QProgressBar {{
                background-color: {Colors.BG_INPUT};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                text-align: center;
                color: {Colors.TEXT_PRIMARY};
            }}
            QProgressBar::chunk {{
                background-color: {Colors.PRIMARY};
                border-radius: 3px;
            }}
        """

    @staticmethod
    def _secondary_label(text):
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 12px; font-weight: 500;")
        return lbl

    # ----------------------------------------------------------
    # closeEvent
    # ----------------------------------------------------------
    def closeEvent(self, event):
        self.stop_all()
        if self._model_loader and self._model_loader.isRunning():
            self._model_loader.terminate()
            self._model_loader.wait(2000)
        if self.worker.isRunning():
            self.worker.stop()
        self.db.flush_tracking_buffer()
        event.accept()

    # ----------------------------------------------------------
    # ROI 鼠标事件
    # ----------------------------------------------------------
    def _label_to_image_coords(self, label_pos):
        """将 QLabel 上的坐标转换为原始图像坐标"""
        pixmap = self.label_display.pixmap()
        if not pixmap or pixmap.isNull() or self.current_res_img is None:
            return None, None

        label_size = self.label_display.size()
        # _display_image 中使用 KeepAspectRatio 缩放，计算实际显示区域
        img_h, img_w = self.current_res_img.shape[:2]
        scaled_w = label_size.width()
        scaled_h = int(img_h * scaled_w / img_w) if img_w > 0 else label_size.height()
        if scaled_h > label_size.height():
            scaled_h = label_size.height()
            scaled_w = int(img_w * scaled_h / img_h) if img_h > 0 else label_size.width()

        ox = (label_size.width() - scaled_w) // 2
        oy = (label_size.height() - scaled_h) // 2

        lx = label_pos.x() - ox
        ly = label_pos.y() - oy
        ix = int(lx * img_w / scaled_w) if scaled_w > 0 else 0
        iy = int(ly * img_h / scaled_h) if scaled_h > 0 else 0
        ix = max(0, min(ix, img_w))
        iy = max(0, min(iy, img_h))
        return ix, iy

    def mouse_press(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self.current_res_img is not None:
            self.is_drawing = True
            self.roi_start_point = event.pos()
            self.roi_end_point = event.pos()

    def mouse_move(self, event: QMouseEvent):
        if self.is_drawing:
            self.roi_end_point = event.pos()
            self._refresh_roi_overlay()

    def mouse_release(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self.is_drawing:
            self.is_drawing = False
            self.roi_end_point = event.pos()
            self._refresh_roi_overlay()

            # 转换为图像坐标并保存警戒区域
            ix1, iy1 = self._label_to_image_coords(self.roi_start_point)
            ix2, iy2 = self._label_to_image_coords(self.roi_end_point)
            if ix1 is None:
                return

            if abs(ix2 - ix1) > 10 and abs(iy2 - iy1) > 10:
                self.alarm_zone = (min(ix1, ix2), min(iy1, iy2), max(ix1, ix2), max(iy1, iy2))
                QMessageBox.information(self, "警戒区域",
                    f"警戒区域已设置!\n坐标: {self.alarm_zone}")
            else:
                self.roi_start_point = None
                self.roi_end_point = None

    def _refresh_roi_overlay(self):
        """在缓存的底图上用 QPainter 绘制矩形，不修改原图"""
        if not self._base_pixmap or self._base_pixmap.isNull():
            return
        overlay = self._base_pixmap.copy()
        painter = QPainter(overlay)
        pen = QPen(QColor(255, 255, 0), 2, Qt.SolidLine)
        painter.setPen(pen)
        rect = QRect(self.roi_start_point, self.roi_end_point).normalized()
        painter.drawRect(rect)
        painter.fillRect(rect, QColor(255, 255, 0, 40))
        painter.end()
        self.label_display.setPixmap(overlay)

    # ----------------------------------------------------------
    # 【修复】定时器 — Worker 不忙时才读新帧提交
    # ----------------------------------------------------------
    def on_timer_tick(self):
        if not self.cap or not self.cap.isOpened():
            return

        # 用户正在拖拽进度条，跳过
        if self._slider_dragging:
            return

        # Worker 还在忙，跳过本 tick，不读帧不丢帧
        if self.worker.is_busy:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_all()
            return

        self._last_frame = frame

        # 更新进度条（拖拽时跳过，避免滑块跳动）
        if not self._slider_dragging:
            cur = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.slider.blockSignals(True)
            self.slider.setMaximum(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            self.slider.setValue(cur)
            self.slider.blockSignals(False)

        # 同步参数
        self.worker.conf_thresh = self.spin_conf.value()
        self.worker.is_tracking = self.check_track.isChecked()
        self.is_tracking_enabled = self.check_track.isChecked()

        # 提交帧
        self.worker.submit_frame(frame.copy())

    # ----------------------------------------------------------
    # Worker 回调
    # ----------------------------------------------------------
    def on_worker_result(self, original_frame, results, processing_time_ms):
        fps = 1000.0 / processing_time_ms if processing_time_ms > 0 else 0
        self.process_stats(results, original_frame, fps, processing_time_ms)

        plot_img = results[0].plot()

        if self.alarm_zone is not None:
            x1, y1, x2, y2 = self.alarm_zone
            cv2.rectangle(plot_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(plot_img, "ALERT ZONE", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        if self.is_tracking_enabled:
            self.trajectory.draw_trajectories(plot_img)

        self.current_res_img = plot_img
        self._display_image(self.current_res_img)
        self.frame_count += 1

    # ----------------------------------------------------------
    # 显示图像
    # ----------------------------------------------------------
    def _display_image(self, img):
        if img is None:
            return
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        scaled = pixmap.scaled(self.label_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.original_pixmap = pixmap
        self._base_pixmap = scaled  # 缓存缩放后的底图
        # 如果正在绘制 ROI，在底图上叠加矩形
        if self.is_drawing and self.roi_start_point and self.roi_end_point:
            overlay = scaled.copy()
            painter = QPainter(overlay)
            pen = QPen(QColor(255, 255, 0), 2, Qt.SolidLine)
            painter.setPen(pen)
            rect = QRect(self.roi_start_point, self.roi_end_point).normalized()
            painter.drawRect(rect)
            painter.fillRect(rect, QColor(255, 255, 0, 40))
            painter.end()
            self.label_display.setPixmap(overlay)
        else:
            self.label_display.setPixmap(scaled)

    # ----------------------------------------------------------
    # 处理统计
    # ----------------------------------------------------------
    def process_stats(self, results, original_frame, fps=0.0, proc_ms=0.0):
        res = results[0]

        is_alarm = False
        alarm_msg = "系统就绪"
        alarm_color = Colors.SUCCESS

        current_positions = []
        behavior_counts = {'stationary': 0, 'walking': 0, 'running': 0}

        if self.alarm_zone is not None and len(res.boxes) > 0:
            xz1, yz1, xz2, yz2 = self.alarm_zone
            for box in res.boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                fx = (bx1 + bx2) // 2; fy = by2
                current_positions.append((fx, fy))
                if xz1 < fx < xz2 and yz1 < fy < yz2:
                    is_alarm = True
                    alarm_msg = "⚠ 警戒区域入侵!"
                    alarm_color = Colors.ERROR
                    # 首次入侵立即记录，之后每30帧记录一次
                    if not self._alarm_was_active or self.frame_count % 30 == 0:
                        self.db.insert_alarm_event({
                            'alarm_type': 'zone_intrusion',
                            'person_count': len(res.boxes),
                            'zone_coordinates': str(self.alarm_zone),
                            'description': f'检测到 {len(res.boxes)} 人进入警戒区域'
                        })
                        self.alarm_log.append(
                            f"[{datetime.now().strftime('%H:%M:%S')}] ⚠ 警报: 警戒区域入侵 (人数: {len(res.boxes)})")
                    break
        self._alarm_was_active = is_alarm

        if self.is_tracking_enabled and current_positions:
            crowded, _ = self.behavior.detect_crowding(current_positions, threshold=5, radius=100)
            if crowded and self.frame_count % 60 == 0:
                self.db.insert_alarm_event({
                    'alarm_type': 'crowding',
                    'person_count': len(current_positions),
                    'zone_coordinates': '',
                    'description': f'检测到人群聚集 (人数: {len(current_positions)})'
                })
                self.alarm_log.append(
                    f"[{datetime.now().strftime('%H:%M:%S')}] ⚠ 人群聚集: 检测到异常聚集 (人数: {len(current_positions)})")

        self.status_indicator.update_status(alarm_msg, alarm_color)
        self.lbl_count.setText(f"🚶 行人总数: {len(res.boxes)}")
        self.lbl_fps.setText(f"📈 FPS: {fps:.1f}")
        self.lbl_speed.setText(f"⏱ 推理耗时: {proc_ms:.1f} ms")

        if self.is_tracking_enabled:
            for box in res.boxes:
                if box.id is not None:
                    tid = int(box.id[0])
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                    cx = (bx1 + bx2) // 2; cy = (by1 + by2) // 2
                    self.trajectory.update(tid, cx, cy)
                    self.heatmap.update_position(cx, cy)
                    bhv = self.behavior.analyze(tid, cx, cy,
                                                original_frame.shape[1], original_frame.shape[0])
                    if bhv in behavior_counts:
                        behavior_counts[bhv] += 1
                    self.db.buffer_tracking_data({
                        'track_id': tid, 'x_center': cx, 'y_center': cy,
                        'x1': bx1, 'y1': by1, 'x2': bx2, 'y2': by2,
                        'confidence': float(box.conf[0]), 'behavior_type': bhv,
                    })
        else:
            # 未开启追踪时，仍采集热力图数据
            for box in res.boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                cx = (bx1 + bx2) // 2; cy = (by1 + by2) // 2
                self.heatmap.update_position(cx, cy)

        self._db_flush_counter += 1
        if self._db_flush_counter >= 60:
            self.db.flush_tracking_buffer()
            self._db_flush_counter = 0

        self.table.setRowCount(0)
        for i, box in enumerate(res.boxes):
            self.table.insertRow(i)
            tid_str = f"ID_{int(box.id[0])}" if (self.is_tracking_enabled and box.id is not None) else f"ID_{i + 1}"
            cls_name = res.names[int(box.cls[0])]
            conf_str = f"{float(box.conf[0]):.2f}"

            if self.is_tracking_enabled and box.id is not None:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                bhv = self.behavior.analyze(int(box.id[0]), (bx1 + bx2) // 2, (by1 + by2) // 2,
                                            original_frame.shape[1], original_frame.shape[0])
                bhv_text = {'stationary': '静止', 'walking': '行走', 'running': '奔跑'}.get(bhv, '未知')
            else:
                bhv_text = '-'

            for col, txt in enumerate((tid_str, cls_name, conf_str, bhv_text)):
                item = QTableWidgetItem(txt)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, col, item)

        self.person_count_history.append(len(res.boxes))
        self.time_history.append(datetime.now().strftime('%H:%M:%S'))
        if len(self.person_count_history) > 50:
            self.person_count_history = self.person_count_history[-50:]
            self.time_history = self.time_history[-50:]

        if self.frame_count % 10 == 0 and self.person_count_history:
            self.chart_widget.plot_line_chart(
                list(range(len(self.person_count_history))), self.person_count_history,
                title="行人数量实时变化", xlabel="帧", ylabel="人数")

        if self.frame_count % 30 == 0 and self.heatmap.heatmap is not None:
            self.heatmap_widget.update_heatmap(self.heatmap.heatmap)

        if self.frame_count % 60 == 0 and self.is_tracking_enabled:
            self._update_behavior_chart(behavior_counts)

    # ----------------------------------------------------------
    # 静态图像检测
    # ----------------------------------------------------------
    def detect_image(self):
        self.stop_all()
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.jpg *.jpeg *.png *.bmp)")
        if not path:
            return

        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
        if img is None:
            self.label_display.setText('⚠ 图片加载失败')
            return

        conf = self.spin_conf.value()
        self.is_tracking_enabled = self.check_track.isChecked()

        try:
            t0 = cv2.getTickCount()
            if self.is_tracking_enabled:
                results = self.model.track(source=img, classes=[0], conf=conf, verbose=False, persist=True)
            else:
                results = self.model.predict(source=img, classes=[0], conf=conf, verbose=False)
            dt = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000

            self.process_stats(results, img, 0, dt)
            self.current_res_img = results[0].plot()
            self._display_image(self.current_res_img)
            self.status_indicator.update_status("检测完成", Colors.INFO)

            self.db.insert_detection_record({
                'source_type': 'image', 'source_path': path,
                'model_name': self.current_model_name,
                'confidence_threshold': conf,
                'tracking_enabled': 1 if self.is_tracking_enabled else 0,
                'total_persons': len(results[0].boxes),
                'fps': 0, 'processing_time': dt,
            })
        except Exception as e:
            QMessageBox.critical(self, "错误", f"检测失败: {e}")

    # ----------------------------------------------------------
    # 打开视频
    # ----------------------------------------------------------
    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "Videos (*.mp4 *.avi *.mov)")
        if not path:
            return
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开视频文件")
            return
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.setMaximum(self.total_frames)

        # 根据视频分辨率重新初始化热力图
        vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if vw > 0 and vh > 0:
            self.heatmap = HeatmapGenerator(width=vw, height=vh)

        # 【修复】确保 Worker 模型已绑定
        self.worker.model = self.model
        self.timer.start(30)
        self.statusBar().showMessage(f"正在播放: {os.path.basename(path)}")

        self.db.insert_detection_record({
            'source_type': 'video', 'source_path': path,
            'model_name': self.current_model_name,
            'confidence_threshold': self.spin_conf.value(),
            'tracking_enabled': 1 if self.check_track.isChecked() else 0,
            'total_persons': 0, 'fps': 0, 'processing_time': 0,
        })

    # ----------------------------------------------------------
    # 打开摄像头
    # ----------------------------------------------------------
    def open_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开摄像头")
            return
        self.slider.setMaximum(0)

        # 根据摄像头分辨率重新初始化热力图
        vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if vw > 0 and vh > 0:
            self.heatmap = HeatmapGenerator(width=vw, height=vh)

        # 【修复】确保 Worker 模型已绑定
        self.worker.model = self.model
        self.timer.start(30)
        self.statusBar().showMessage("正在使用摄像头")

    # ----------------------------------------------------------
    # 播放/暂停
    # ----------------------------------------------------------
    def toggle_play_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.timer.stop()
            self.btn_play_pause.setText("▶ 播放")
        else:
            self.timer.start(30)
            self.btn_play_pause.setText("⏸ 暂停")

    # ----------------------------------------------------------
    # 进度条拖拽
    # ----------------------------------------------------------
    def _on_slider_pressed(self):
        self._slider_dragging = True
        self.timer.stop()

    def _on_slider_moved(self, value):
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, value)
            ret, frame = self.cap.read()
            if ret:
                self._display_image(frame)

    def _on_slider_released(self):
        self._slider_dragging = False
        if self.cap and self.cap.isOpened():
            pos = self.slider.value()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            if not self.is_paused:
                self.timer.start(30)

    # ----------------------------------------------------------
    # 停止
    # ----------------------------------------------------------
    def stop_all(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.cap = None
        self._slider_dragging = False
        self.db.flush_tracking_buffer()
        self.frame_count = 0
        self.person_count_history.clear()
        self.time_history.clear()
        self.trajectory.clear()
        self.behavior.clear()
        self.heatmap.reset()
        self.heatmap_widget.update_heatmap(self.heatmap.heatmap)
        self.alarm_zone = None
        self.roi_start_point = None
        self.roi_end_point = None
        self.is_paused = False
        self._last_frame = None
        self.label_display.setText('📷 请打开图片或视频开始检测')
        self.btn_play_pause.setText("▶ 播放")
        self.statusBar().showMessage("就绪")

    # ----------------------------------------------------------
    # 模型切换
    # ----------------------------------------------------------
    def change_model(self, text):
        if not text:
            return
        model_name = text.split()[0]  # "yolov8n.pt (轻量)" -> "yolov8n.pt"
        if model_name == self.current_model_name:
            return
        # 取消上一次未完成的加载
        if self._model_loader and self._model_loader.isRunning():
            self._model_loader.finished.disconnect()
            self._model_loader.error.disconnect()
            self._model_loader.terminate()
            self._model_loader.wait(1000)
        # 禁用下拉框防止重复触发，显示加载状态
        self.combo_model.setEnabled(False)
        self.status_indicator.update_status(f"正在加载模型 {model_name}...", Colors.WARNING)
        self.statusBar().showMessage(f"正在加载模型 {model_name}...")
        # 后台线程加载模型
        self._model_loader = ModelLoadWorker(model_name)
        self._model_loader.finished.connect(self._on_model_loaded)
        self._model_loader.error.connect(self._on_model_load_error)
        self._model_loader.start()

    def _on_model_loaded(self, model, model_name):
        self.model = model
        self.worker.model = self.model
        self.current_model_name = model_name
        self.combo_model.setEnabled(True)
        self.status_indicator.update_status(f"模型 {model_name} 已就绪", Colors.SUCCESS)
        self.statusBar().showMessage(f"模型 {model_name} 加载成功")

    def _on_model_load_error(self, error_msg):
        self.combo_model.setEnabled(True)
        self.status_indicator.update_status("模型加载失败", Colors.ERROR)
        QMessageBox.critical(self, "错误", f"模型切换失败: {error_msg}")

    # ----------------------------------------------------------
    # 离线视频导出
    # ----------------------------------------------------------
    def export_video(self):
        src_path, _ = QFileDialog.getOpenFileName(
            self, "选择源视频", "", "Videos (*.mp4 *.avi *.mov)")
        if not src_path:
            return

        dst_path, _ = QFileDialog.getSaveFileName(
            self, "保存导出视频", "", "MP4 Files (*.mp4)")
        if not dst_path:
            return

        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开源视频")
            return

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(dst_path, fourcc, fps_in, (w, h))

        conf = self.spin_conf.value()
        use_track = self.check_track.isChecked()

        progress = QProgressDialog("正在导出视频...", "取消", 0, total, self)
        progress.setWindowTitle("视频导出")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        frame_idx = 0
        try:
            while True:
                if progress.wasCanceled():
                    break
                ret, frame = cap.read()
                if not ret:
                    break

                if use_track:
                    results = self.model.track(source=frame, classes=[0], conf=conf,
                                               verbose=False, persist=True)
                else:
                    results = self.model.predict(source=frame, classes=[0], conf=conf,
                                                 verbose=False)

                annotated = results[0].plot()
                if annotated.shape[:2] != (h, w):
                    annotated = cv2.resize(annotated, (w, h))
                writer.write(annotated)

                frame_idx += 1
                progress.setValue(frame_idx)

                if frame_idx % 5 == 0:
                    QApplication.processEvents()
        finally:
            cap.release()
            writer.release()

        progress.setValue(total)
        QMessageBox.information(self, "完成", f"视频导出完成!\n共处理 {frame_idx} 帧\n保存至: {dst_path}")

    # ----------------------------------------------------------
    # 数据导出
    # ----------------------------------------------------------
    def export_data(self):
        try:
            if HAS_PANDAS:
                path, _ = QFileDialog.getSaveFileName(
                    self, "导出数据", "", "Excel Files (*.xlsx);;CSV Files (*.csv)")
                if not path:
                    return
                if path.endswith('.xlsx'):
                    ok = self.db.export_to_excel(path)
                    if ok:
                        QMessageBox.information(self, "成功", f"数据已导出:\n{path}")
                    else:
                        QMessageBox.warning(self, "警告", "导出失败，请确认 pandas 和 openpyxl 已安装")
                else:
                    self.db.export_to_csv("detection_records", path)
                    QMessageBox.information(self, "成功", f"数据已导出:\n{path}")
            else:
                path, _ = QFileDialog.getSaveFileName(
                    self, "导出数据", "", "CSV Files (*.csv)")
                if path:
                    self.db.export_to_csv("detection_records", path)
                    QMessageBox.information(self, "成功", f"CSV 已导出:\n{path}\n提示: 安装 pandas 可导出 Excel 格式")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"导出数据时发生错误:\n{str(e)}")

    # ----------------------------------------------------------
    # 热力图清除
    # ----------------------------------------------------------
    def clear_heatmap(self):
        self.heatmap.reset()
        blank = np.zeros((self.heatmap.height, self.heatmap.width), dtype=np.float32)
        self.heatmap_widget.update_heatmap(blank)
        QMessageBox.information(self, "成功", "热力图已清除")

    # ----------------------------------------------------------
    # 报警日志
    # ----------------------------------------------------------
    def _load_alarm_history(self):
        """从数据库加载历史报警记录到日志控件"""
        try:
            events = self.db.get_alarm_events()
            for ev in events:
                # ev: (id, timestamp, alarm_type, person_count, zone_coordinates, snapshot_path, description)
                ts = ev[1] if ev[1] else ""
                alarm_type = ev[2] or ""
                desc = ev[6] or ""
                person_count = ev[3] or 0
                if alarm_type == 'zone_intrusion':
 

# ============================================================
# 启动入口
# ============================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = YoloVideoPro()
    window.show()
    sys.exit(app.exec_())
