import sys
import os
import numpy as np
import librosa
import pyqtgraph as pg
import pyqtgraph.exporters as pg_exporters  # [Phase 3-3] エクスポート機能
import sounddevice as sd
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QVBoxLayout,
    QWidget,
    QMenuBar,
    QPushButton,
    QHBoxLayout,
    QDialog,
    QComboBox,
    QSpinBox,
    QFormLayout,
    QDialogButtonBox,
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import QLocale, QTimer


class PySurferWindow(QMainWindow):
    """
    メインウィンドウクラス
    """

    def __init__(self):
        super().__init__()
        # [Phase 3-3] タイトル変更
        self.setWindowTitle("PySurfer (フェーズ 3-3)")
        self.setGeometry(100, 100, 800, 600)

        # (Model, STFTピラミッド, Timer の定義は変更なし)
        # ... (省略) ...
        self.audio_data = None
        self.sample_rate = None
        self.duration = 0.0
        self.nyquist = 0.0
        self.current_filename = None
        self.stft_level_params = [
            {"threshold": 5.0, "hop": 512, "n_fft": 2048},
            {"threshold": 1.0, "hop": 256, "n_fft": 1024},
            {"threshold": 0.0, "hop": 128, "n_fft": 512},
        ]
        self.stft_levels_data = []
        self.current_stft_level = -1
        self.zoom_timer = QTimer(self)
        self.zoom_timer.setSingleShot(True)
        self.zoom_timer.setInterval(150)
        self.zoom_timer.timeout.connect(self.update_spectrogram_view)

        # --- View (UIのセットアップ) ---
        self.init_ui()

    def init_ui(self):
        """
        UIの初期化を行います
        """
        # メニューバーの作成
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&ファイル")

        # "File" メニューに "Open" アクションを追加
        open_action = QAction("&音声ファイルを開く...", self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        # [Phase 3-3] "画像として保存" アクションの追加
        file_menu.addSeparator()  # 区切り線

        self.export_wave_action = QAction("&波形を画像として保存...", self)
        self.export_wave_action.triggered.connect(self.export_waveform_image)
        file_menu.addAction(self.export_wave_action)

        self.export_spec_action = QAction("&スペクトログラムを画像として保存...", self)
        self.export_spec_action.triggered.connect(self.export_spec_image)
        file_menu.addAction(self.export_spec_action)

        # (ボタン、レイアウト、各ペインの設定は変更なし)
        # ... (省略) ...
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        control_layout = QHBoxLayout()
        self.play_button = QPushButton("▶ 再生")
        self.stop_button = QPushButton("■ 停止")
        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addStretch()
        main_layout.addLayout(control_layout)
        self.plot_widget = pg.PlotWidget()
        main_layout.addWidget(self.plot_widget, stretch=1)
        self.plot_widget.setBackground("w")
        self.plot_widget.setLabel("left", "振幅")
        self.plot_widget.setLabel("bottom", "時間 (秒)")
        self.plot_widget.showGrid(x=True, y=True)
        self.spec_widget = pg.PlotWidget()
        main_layout.addWidget(self.spec_widget, stretch=2)
        self.spec_widget.setBackground("w")
        self.spec_widget.setLabel("left", "周波数 (Hz)")
        self.spec_widget.setLabel("bottom", "時間 (秒)")
        self.spec_image = pg.ImageItem()
        self.spec_widget.addItem(self.spec_image)
        pos = np.array([0.0, 1.0])
        color = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        lut = cmap.getLookupTable(0.0, 1.0, 256)
        self.spec_image.setLookupTable(lut)
        self.plot_widget.setMouseEnabled(x=True, y=False)
        self.view_box = self.plot_widget.getPlotItem().getViewBox()
        self.view_box.setMouseMode(pg.ViewBox.PanMode)
        self.spec_widget.setMouseEnabled(x=True, y=False)
        self.spec_view_box = self.spec_widget.getPlotItem().getViewBox()
        self.spec_view_box.setMouseMode(pg.ViewBox.PanMode)
        self.view_box.setXLink(self.spec_view_box)
        self.view_box.sigRangeChanged.connect(self.on_range_changed)
        self.play_button.clicked.connect(self.play_audio)
        self.stop_button.clicked.connect(self.stop_audio)

        # [Phase 3-3] 初期状態ではボタンとメニューを無効化
        self.play_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.export_wave_action.setEnabled(False)
        self.export_spec_action.setEnabled(False)

    # --- Controller (イベント処理) ---

    def open_file(self):
        """
        "Open" アクションが実行されたときの処理
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "音声ファイルを開く", "", "音声ファイル (*.wav *.mp3 *.flac)")

        if file_path:
            # [変更] ファイル名をモデルに保存
            try:
                # ファイルパスからベース名 (例: "sound.wav") を取得
                base_name = os.path.basename(file_path)
                # ベース名から拡張子を除いた部分 (例: "sound") を保存
                self.current_filename, _ = os.path.splitext(base_name)
                print(f"現在のファイル名を設定: {self.current_filename}")
            except Exception as e:
                print(f"ファイル名の処理エラー: {e}")
                self.current_filename = None  # エラー時はリセット
            print(f"ファイルが選択されました: {file_path}")

            try:
                # (読み込み、STFTピラミッド計算は変更なし)
                # ... (省略) ...
                self.stop_audio()
                self.current_stft_level = -1
                self.audio_data, self.sample_rate = librosa.load(file_path, sr=None)
                self.duration = len(self.audio_data) / self.sample_rate
                self.nyquist = self.sample_rate / 2.0
                print(f"読み込み完了: {self.duration:.2f} 秒, SR: {self.sample_rate} Hz")
                self.spec_widget.setYRange(0, self.nyquist)
                self.calculate_stft_pyramid()
                self.plot_waveform()
                self.update_spectrogram_view(force_update=True)

                # [Phase 3-3] ファイル読み込み成功時にボタンとメニューを有効化
                self.play_button.setEnabled(True)
                self.stop_button.setEnabled(True)
                self.export_wave_action.setEnabled(True)
                self.export_spec_action.setEnabled(True)

            except Exception as e:
                print(f"ファイルの読み込みエラー: {e}")
                # [Phase 3-3] 失敗時は無効化
                self.play_button.setEnabled(False)
                self.stop_button.setEnabled(False)
                self.export_wave_action.setEnabled(False)
                self.export_spec_action.setEnabled(False)

    # (plot_waveform, calculate_stft_pyramid, on_range_changed, update_spectrogram_view は変更なし)
    # ... (省略) ...
    def plot_waveform(self):
        if self.audio_data is None:
            return
        self.plot_widget.clear()
        time_axis = np.linspace(0, self.duration, num=len(self.audio_data))
        data_to_plot = self.audio_data
        if self.audio_data.ndim > 1:
            data_to_plot = self.audio_data[0]
        self.plot_widget.plot(time_axis, data_to_plot, pen=pg.mkPen(color=(0, 0, 200), width=1))
        print("波形を描画しました。")
        self.view_box.setLimits(xMin=0, xMax=self.duration, yMin=None, yMax=None)
        self.spec_view_box.setLimits(xMin=0, xMax=self.duration, yMin=None, yMax=self.nyquist)
        self.plot_widget.autoRange()

    def calculate_stft_pyramid(self):
        print("STFTピラミッドを計算中...")
        self.stft_levels_data = []
        data_to_calc = self.audio_data
        if self.audio_data.ndim > 1:
            data_to_calc = self.audio_data[0]
        for i, params in enumerate(self.stft_level_params):
            print(f"  レベル {i} (hop={params['hop']}, n_fft={params['n_fft']}) を計算中...")
            D = np.abs(librosa.stft(data_to_calc, n_fft=params["n_fft"], hop_length=params["hop"]))
            S_db = librosa.amplitude_to_db(D, ref=np.max)
            self.stft_levels_data.append(S_db)
        print("STFTピラミッドの計算が完了しました。")

    def on_range_changed(self):
        if self.zoom_timer.isActive():
            self.zoom_timer.stop()
        self.zoom_timer.start()

    def update_spectrogram_view(self, force_update=False):
        if not self.stft_levels_data:
            return
        x_min, x_max = self.view_box.viewRange()[0]
        visible_duration = x_max - x_min
        target_level_index = 0
        for i, params in enumerate(self.stft_level_params):
            if visible_duration <= params["threshold"]:
                target_level_index = i
            else:
                if i > 0:
                    target_level_index = i - 1
                break
        if not force_update and target_level_index == self.current_stft_level:
            return
        print(f"表示を STFT レベル {target_level_index} に切り替えます (表示時間: {visible_duration:.2f}秒)")
        self.current_stft_level = target_level_index
        S_db = self.stft_levels_data[target_level_index]
        self.spec_image.setImage(S_db.T)
        self.spec_image.setRect(0, 0, self.duration, self.nyquist)

    # --- [Phase 3-3] Controller (Image Export) ---

    def export_waveform_image(self):
        """
        波形ペインを画像として保存する
        """
        self._export_plot_widget(self.plot_widget, "波形")

    def export_spec_image(self):
        """
        スペクトログラムペインを画像として保存する
        """
        self._export_plot_widget(self.spec_widget, "スペクトログラム")

    def _export_plot_widget(self, widget: pg.PlotWidget, name: str):
        """
        指定されたPlotWidgetをエクスポートする共通メソッド
        """
        print(f"{name} のエクスポート処理を開始します...")

        # [変更] デフォルトのファイル名を動的に生成
        if self.current_filename:
            # 例: "my_audio_file_波形.png"
            default_filename = f"{self.current_filename}_{name}.png"
        else:
            # ファイル名が取得できなかった場合のフォールバック
            default_filename = f"{name}_export.png"

        # 1. ファイルダイアログを表示
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            f"{name} を画像として保存",
            default_filename,  # デフォルトのファイル名
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg)",  # フィルタ
        )

        if not file_path:
            print("エクスポートがキャンセルされました。")
            return

        # 2. pyqtgraph.exporters を使って保存
        try:
            # エクスポート対象のPlotItemを取得
            exporter = pg_exporters.ImageExporter(widget.getPlotItem())

            # ファイルにエクスポート
            exporter.export(file_path)

            print(f"画像を {file_path} に保存しました。")

        except Exception as e:
            print(f"画像のエクスポートに失敗しました: {e}")

    # --- Controller (Audio Playback) ---
    # (play_audio, stop_audio は変更なし)
    # ... (省略) ...
    def play_audio(self):
        if self.audio_data is not None:
            print("音声を再生します...")
            try:
                sd.play(self.audio_data, self.sample_rate)
            except Exception as e:
                print(f"再生エラー: {e}")

    def stop_audio(self):
        print("再生を停止します。")
        sd.stop()


# --- アプリケーションの実行 ---
if __name__ == "__main__":
    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    QLocale.setDefault(QLocale())
    window = PySurferWindow()
    window.show()
    sys.exit(app.exec())
