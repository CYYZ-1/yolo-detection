import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                             QHBoxLayout, QWidget, QLabel, QFileDialog, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QGroupBox, QSlider, 
                             QAbstractItemView, QComboBox, QDoubleSpinBox, QMessageBox, QProgressDialog)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from ultralytics import YOLO

class YoloVideoPro(QMainWindow):
    def __init__(self):
        super().__init__()
        # 默认加载模型
        self.current_model_name = 'yolov8n.pt'
        self.model = YOLO(self.current_model_name) 
        
        # 状态变量
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_paused = False
        self.total_frames = 0
        self.current_res_img = None # 用于保存当前处理后的图像

        self.initUI()

    def initUI(self):
        self.setWindowTitle('基于YOLO的图像行人检测系统 - 毕业设计完整版')
        self.setGeometry(100, 100, 1300, 850)
        
        # 全局现代暗色 QSS 样式表 (新增了下拉框和微调框的样式)
        self.setStyleSheet("""
            QMainWindow { background-color: #1E1E2E; }
            QLabel { color: #CDD6F4; font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; font-size: 14px; }
            QGroupBox { color: #89B4FA; font-weight: bold; border: 1px solid #45475A; border-radius: 8px; margin-top: 15px; font-size: 15px; }
            QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 0 5px; }
            QPushButton { background-color: #313244; color: #CDD6F4; border: 1px solid #45475A; border-radius: 6px; padding: 8px 12px; font-weight: bold; font-size: 13px; }
            QPushButton:hover { background-color: #45475A; border: 1px solid #89B4FA; color: #89B4FA; }
            QPushButton:pressed { background-color: #89B4FA; color: #1E1E2E; }
            QPushButton#btn_stop { color: #F38BA8; }
            QPushButton#btn_stop:hover { border: 1px solid #F38BA8; background-color: #45475A; }
            QPushButton#btn_stop:pressed { background-color: #F38BA8; color: #1E1E2E; }
            QPushButton#btn_save { color: #A6E3A1; }
            QPushButton#btn_save:hover { border: 1px solid #A6E3A1; background-color: #45475A; }
            QSlider::groove:horizontal { height: 6px; background: #313244; border-radius: 3px; }
            QSlider::sub-page:horizontal { background: #89B4FA; border-radius: 3px; }
            QSlider::handle:horizontal { background: #FFFFFF; width: 16px; margin-top: -5px; margin-bottom: -5px; border-radius: 8px; }
            QSlider::handle:horizontal:hover { background: #89B4FA; width: 18px; margin-top: -6px; margin-bottom: -6px; border-radius: 9px; }
            QTableWidget { background-color: #1E1E2E; alternate-background-color: #252535; color: #CDD6F4; border: 1px solid #45475A; border-radius: 8px; gridline-color: #313244; selection-background-color: #89B4FA; selection-color: #1E1E2E; font-size: 13px; }
            QHeaderView::section { background-color: #313244; color: #89B4FA; padding: 6px; border: none; font-weight: bold; font-size: 14px; }
            QComboBox, QDoubleSpinBox { background-color: #313244; color: #CDD6F4; border: 1px solid #45475A; border-radius: 4px; padding: 5px; font-size: 13px; }
            QComboBox:hover, QDoubleSpinBox:hover { border: 1px solid #89B4FA; }
        """)

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # ================= 左侧：画面与进度控制 =================
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.label_display = QLabel('请加载图片或视频进行行人检测...')
        self.label_display.setAlignment(Qt.AlignCenter)
        self.label_display.setStyleSheet("background-color: #11111B; border: 2px solid #313244; border-radius: 10px; font-size: 20px; color: #6C7086;")
        self.label_display.setScaledContents(True)
        left_layout.addWidget(self.label_display, stretch=10)

        # 进度条
        slider_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setCursor(Qt.PointingHandCursor)
        self.slider.setEnabled(False)
        self.slider.sliderPressed.connect(self.pause_video)
        self.slider.sliderReleased.connect(self.resume_video)
        self.slider.valueChanged.connect(self.set_video_pos)
        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setFixedWidth(100)
        self.lbl_time.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.lbl_time)
        left_layout.addLayout(slider_layout)

        # 播放控制按钮
        btn_layout = QHBoxLayout()
        self.btn_img = QPushButton('🖼️ 选图分析')
        self.btn_video = QPushButton('🎬 视频播放检测')
        self.btn_play_pause = QPushButton('⏸ 暂停')
        self.btn_stop = QPushButton('⏹ 停止')
        self.btn_stop.setObjectName("btn_stop")
        
        for btn in [self.btn_img, self.btn_video, self.btn_play_pause, self.btn_stop]:
            btn.setCursor(Qt.PointingHandCursor)

        btn_layout.addWidget(self.btn_img)
        btn_layout.addWidget(self.btn_video)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_play_pause)
        btn_layout.addWidget(self.btn_stop)
        left_layout.addLayout(btn_layout)
        left_widget.setLayout(left_layout)

        # ================= 右侧：参数面板 =================
        right_widget = QWidget()
        right_widget.setFixedWidth(320)
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # 1. 毕设要求：检测设置面板
        settings_group = QGroupBox("⚙️ 检测参数设置")
        settings_vbox = QVBoxLayout()
        settings_vbox.setSpacing(10)
        
        # 模型选择
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("选择模型:"))
        self.combo_model = QComboBox()
        self.combo_model.addItems(["yolov8n.pt", "yolov5nu.pt", "yolov8s.pt"]) # YOLOv5 官方推荐使用 yolov5nu.pt
        self.combo_model.currentTextChanged.connect(self.change_model)
        model_layout.addWidget(self.combo_model)
        
        # 置信度阈值
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("置信度阈值:"))
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.1, 1.0)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.5) # 默认 0.5
        conf_layout.addWidget(self.spin_conf)
        
        settings_vbox.addLayout(model_layout)
        settings_vbox.addLayout(conf_layout)
        settings_group.setLayout(settings_vbox)
        right_layout.addWidget(settings_group)
        
        # 2. 毕设要求：导出与保存面板
        export_group = QGroupBox("💾 结果导出")
        export_vbox = QVBoxLayout()
        export_vbox.setSpacing(10)
        self.btn_save_img = QPushButton('📥 保存当前画面(图)')
        self.btn_save_img.setObjectName("btn_save")
        self.btn_export_video = QPushButton('🎞️ 离线导出检测视频')
        self.btn_export_video.setObjectName("btn_save")
        self.btn_save_img.clicked.connect(self.save_current_image)
        self.btn_export_video.clicked.connect(self.export_video_file)
        export_vbox.addWidget(self.btn_save_img)
        export_vbox.addWidget(self.btn_export_video)
        export_group.setLayout(export_vbox)
        right_layout.addWidget(export_group)

        # 3. 实时参数组
        stats_group = QGroupBox("📊 实时检测统计")
        stats_vbox = QVBoxLayout()
        self.lbl_class_filter = QLabel("🎯 类别过滤: 仅限行人(Person)")
        self.lbl_class_filter.setStyleSheet("color: #F38BA8; font-size: 13px;")
        self.lbl_count = QLabel("🚶 行人总数: 0")
        self.lbl_count.setStyleSheet("font-size: 16px; font-weight: bold; color: #A6E3A1; margin-top: 5px;")
        self.lbl_speed = QLabel("⚡ 耗时: 0.0 ms")
        self.lbl_speed.setStyleSheet("font-size: 14px; color: #F9E2AF; margin-top: 5px;")
        stats_vbox.addWidget(self.lbl_class_filter)
        stats_vbox.addWidget(self.lbl_count)
        stats_vbox.addWidget(self.lbl_speed)
        stats_group.setLayout(stats_vbox)
        right_layout.addWidget(stats_group)
        
        # 数据表格
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(['目标 ID', '置信度'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        right_layout.addWidget(self.table)
        right_widget.setLayout(right_layout)

        main_layout.addWidget(left_widget, stretch=4)
        main_layout.addWidget(right_widget, stretch=1)
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.btn_img.clicked.connect(self.detect_image)
        self.btn_video.clicked.connect(self.open_video)
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        self.btn_stop.clicked.connect(self.stop_all)

    # ================= 核心业务逻辑 =================
    def change_model(self, model_name):
        """切换 YOLO 模型"""
        self.stop_all()
        self.label_display.setText(f'正在加载模型 {model_name}，请稍候...')
        QApplication.processEvents() # 强制刷新UI
        try:
            self.model = YOLO(model_name)
            self.label_display.setText(f'✅ 模型 {model_name} 加载成功！\n等待输入。')
        except Exception as e:
            QMessageBox.critical(self, "模型加载失败", f"无法加载模型：{str(e)}")
            self.label_display.setText('❌ 模型加载失败')

    def save_current_image(self):
        """保存当前带有检测框的图像"""
        if self.current_res_img is None:
            QMessageBox.warning(self, "提示", "当前没有可保存的检测画面！")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存图片", "detect_result.jpg", "Images (*.jpg *.png)")
        if path:
            # OpenCV 默认是 BGR，保存不需要转换
            cv2.imencode('.jpg', self.current_res_img)[1].tofile(path) # 支持中文路径保存
            QMessageBox.information(self, "成功", "图片保存成功！")

    def export_video_file(self):
        """离线处理并导出完整视频"""
        self.stop_all()
        in_path, _ = QFileDialog.getOpenFileName(self, "选择要处理的视频", "", "Videos (*.mp4 *.avi)")
        if not in_path: return
        
        out_path, _ = QFileDialog.getSaveFileName(self, "选择保存位置", "output_detected.mp4", "MP4 Video (*.mp4)")
        if not out_path: return

        cap = cv2.VideoCapture(in_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 初始化 VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        # 进度条对话框
        progress = QProgressDialog("正在逐帧处理并导出视频...", "取消", 0, total_frames, self)
        progress.setWindowTitle("导出进度")
        progress.setWindowModality(Qt.WindowModal)

        conf_thresh = self.spin_conf.value()
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or progress.wasCanceled():
                break
                
            # 推理：classes=[0] 仅行人，根据UI设置的置信度阈值
            results = self.model.predict(source=frame, classes=[0], conf=conf_thresh, verbose=False)
            res_frame = results[0].plot()
            writer.write(res_frame)
            
            frame_idx += 1
            if frame_idx % 5 == 0: # 每5帧更新一次UI以防止卡死
                progress.setValue(frame_idx)
                QApplication.processEvents()

        cap.release()
        writer.release()
        progress.setValue(total_frames)
        
        if not progress.wasCanceled():
            QMessageBox.information(self, "完成", f"视频处理完毕！已保存至:\n{out_path}")

    # ================= 实时播放逻辑 =================
    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择播放视频", "", "Videos (*.mp4 *.avi *.mkv)")
        if path:
            self.cap = cv2.VideoCapture(path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.setRange(0, self.total_frames - 1)
            self.slider.setEnabled(True)
            self.timer.start(30) # 控制播放速度
            self.is_paused = False
            self.btn_play_pause.setText("⏸ 暂停")

    def toggle_play_pause(self):
        if self.is_paused: self.resume_video()
        else: self.pause_video()

    def pause_video(self):
        self.timer.stop()
        self.is_paused = True
        self.btn_play_pause.setText("▶️ 播放")

    def resume_video(self):
        if self.cap:
            self.timer.start(30)
            self.is_paused = False
            self.btn_play_pause.setText("⏸ 暂停")

    def set_video_pos(self):
        if self.cap and self.slider.isSliderDown():
            pos = self.slider.value()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            self.update_frame(manual=True)

    def update_frame(self, manual=False):
        if not self.cap: return
        ret, frame = self.cap.read()
        if ret:
            if not manual:
                current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.slider.blockSignals(True)
                self.slider.setValue(current_frame)
                self.slider.blockSignals(False)
                self.update_time_label(current_frame)

            # 获取 UI 设置的置信度
            conf_thresh = self.spin_conf.value()
            results = self.model.predict(source=frame, classes=[0], conf=conf_thresh, verbose=False)
            
            self.process_stats(results)
            self.current_res_img = results[0].plot() # 缓存当前帧用于保存
            self.display_image(self.current_res_img)
        else:
            if not manual: self.stop_all()

    def detect_image(self):
        self.stop_all()
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.jpg *.jpeg *.png *.bmp)")
        if not path: return
            
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
        if img is None:
            self.label_display.setText('❌ 图片加载失败')
            return

        conf_thresh = self.spin_conf.value()
        results = self.model.predict(source=img, classes=[0], conf=conf_thresh, verbose=False)
        self.process_stats(results)
        
        self.current_res_img = results[0].plot()
        self.display_image(self.current_res_img)
        
        self.lbl_time.setText("图片 / 静态")
        self.btn_play_pause.setText("⏸ 暂停")

    def update_time_label(self, current_frame):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            curr_sec = int(current_frame / fps)
            total_sec = int(self.total_frames / fps)
            self.lbl_time.setText(f"{curr_sec//60:02d}:{curr_sec%60:02d} / {total_sec//60:02d}:{total_sec%60:02d}")

    def process_stats(self, results):
        res = results[0]
        self.lbl_count.setText(f"🚶 行人总数: {len(res.boxes)}")
        self.lbl_speed.setText(f"⚡ 耗时: {sum(res.speed.values()):.1f} ms")
        self.table.setRowCount(0)
        for i, box in enumerate(res.boxes):
            self.table.insertRow(i)
            item_id = QTableWidgetItem(f"Person_{i+1}")
            item_id.setTextAlignment(Qt.AlignCenter)
            item_conf = QTableWidgetItem(f"{float(box.conf[0]):.2f}")
            item_conf.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 0, item_id)
            self.table.setItem(i, 1, item_conf)

    def display_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label_display.setPixmap(QPixmap.fromImage(q_img))

    def stop_all(self):
        self.timer.stop()
        if self.cap: self.cap.release()
        self.slider.setValue(0)
        self.slider.setEnabled(False)
        self.btn_play_pause.setText("⏸ 暂停")
        self.current_res_img = None
        # self.label_display.setText('等待加载视频或图片...') # 停止时保留最后一帧画面更符合用户习惯

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = YoloVideoPro()
    win.show()
    sys.exit(app.exec_())