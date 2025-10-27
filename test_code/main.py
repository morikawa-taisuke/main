import sys
import numpy as np
import librosa
import pyqtgraph as pg
import sounddevice as sd
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout, QWidget, QMenuBar, QPushButton, QHBoxLayout
from PyQt6.QtGui import QAction
from PyQt6.QtCore import QLocale  # [Phase 2 改] 日本語化のためにインポート


class PySurferWindow(QMainWindow):
    """
    メインウィンドウクラス
    """

    def __init__(self):
        super().__init__()
        # [Phase 2 改] 日本語化
        self.setWindowTitle("PySurfer (フェーズ 2)")
        self.setGeometry(100, 100, 800, 400)

        # --- Model (データを保持する場所) ---
        self.audio_data = None
        self.sample_rate = None

        # --- View (UIのセットアップ) ---
        self.init_ui()

    def init_ui(self):
        """
        UIの初期化を行います
        """
        # メニューバーの作成
        menu_bar = self.menuBar()
        # [Phase 2 改] 日本語化
        file_menu = menu_bar.addMenu("&ファイル")

        # "File" メニューに "Open" アクションを追加
        # [Phase 2 改] 日本語化
        open_action = QAction("&音声ファイルを開く...", self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        # メインとなるウィジェット
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 全体のレイアウト (垂直)
        main_layout = QVBoxLayout(central_widget)

        # --- 操作ボタン用のレイアウト (水平) ---
        control_layout = QHBoxLayout()
        # [Phase 2 改] 日本語化
        self.play_button = QPushButton("▶ 再生")
        self.stop_button = QPushButton("■ 停止")

        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addStretch()

        main_layout.addLayout(control_layout)

        # --- プロットウィジェット ---
        self.plot_widget = pg.PlotWidget()
        main_layout.addWidget(self.plot_widget)

        # --- プロットウィジェットの設定 ---
        self.plot_widget.setBackground("w")
        # [Phase 2 改] 日本語化
        self.plot_widget.setLabel("left", "振幅")
        self.plot_widget.setLabel("bottom", "時間 (秒)")
        self.plot_widget.showGrid(x=True, y=True)

        # --- マウス操作の設定 ---
        self.plot_widget.setMouseEnabled(x=True, y=False)
        self.view_box = self.plot_widget.getPlotItem().getViewBox()  # [Phase 2 改] ViewBoxを後で使うため変数に
        self.view_box.setMouseMode(pg.ViewBox.PanMode)

        # --- ボタンの動作を接続 ---
        self.play_button.clicked.connect(self.play_audio)
        self.stop_button.clicked.connect(self.stop_audio)

        self.play_button.setEnabled(False)
        self.stop_button.setEnabled(False)

    # --- Controller (イベント処理) ---

    def open_file(self):
        """
        "Open" アクションが実行されたときの処理
        """
        print("ファイルダイアログを開きます...")

        # [Phase 2 改] 日本語化
        file_path, _ = QFileDialog.getOpenFileName(self, "音声ファイルを開く", "", "音声ファイル (*.wav *.mp3 *.flac)")

        if file_path:
            print(f"ファイルが選択されました: {file_path}")

            try:
                self.stop_audio()

                self.audio_data, self.sample_rate = librosa.load(file_path, sr=None)
                print(f"読み込み完了: {len(self.audio_data)} samples, SR: {self.sample_rate} Hz")

                self.plot_waveform()

                self.play_button.setEnabled(True)
                self.stop_button.setEnabled(True)

            except Exception as e:
                print(f"ファイルの読み込みエラー: {e}")
                self.play_button.setEnabled(False)
                self.stop_button.setEnabled(False)

    def plot_waveform(self):
        """
        Modelのデータ (self.audio_data) を View (plot_widget) に描画する
        """
        if self.audio_data is None:
            return

        self.plot_widget.clear()

        duration = len(self.audio_data) / self.sample_rate
        time_axis = np.linspace(0, duration, num=len(self.audio_data))

        if self.audio_data.ndim > 1:
            data_to_plot = self.audio_data[0]
        else:
            data_to_plot = self.audio_data

        self.plot_widget.plot(time_axis, data_to_plot, pen=pg.mkPen(color=(0, 0, 200), width=1))

        print("波形を描画しました。")

        # --- [Phase 2 改] ズーム範囲の制限 ---
        # setLimits(xMin, xMax, yMin, yMax)
        # X軸 (時間) は 0秒 から 音源の最大長 (duration) まで
        # Y軸は制限なし (None)
        self.view_box.setLimits(xMin=0, xMax=duration, yMin=None, yMax=None)

        # 描画後、一度ビュー全体を表示する
        self.plot_widget.autoRange()

    # --- Controller (Audio Playback) ---

    def play_audio(self):
        """
        Playボタンが押されたときの処理
        """
        if self.audio_data is not None and self.sample_rate is not None:
            print("音声を再生します...")
            try:
                sd.play(self.audio_data, self.sample_rate)
            except Exception as e:
                print(f"再生エラー: {e}")

    def stop_audio(self):
        """
        Stopボタンが押されたときの処理
        """
        print("再生を停止します。")
        sd.stop()


# --- アプリケーションの実行 ---
if __name__ == "__main__":
    pg.setConfigOptions(antialias=True)

    app = QApplication(sys.argv)

    # [Phase 2 改] ロケールをシステムデフォルト（通常は日本語）に設定
    # これにより QFileDialog などが日本語表示になります
    QLocale.setDefault(QLocale())

    window = PySurferWindow()
    window.show()
    sys.exit(app.exec())
