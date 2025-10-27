import sys
import numpy as np
import librosa
import pyqtgraph as pg
import sounddevice as sd
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout, QWidget, QMenuBar, QPushButton, QHBoxLayout
from PyQt6.QtGui import QAction
from PyQt6.QtCore import QLocale, QTimer  # [Phase 3-2] QTimerをインポート


class PySurferWindow(QMainWindow):
    """
    メインウィンドウクラス
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PySurfer (フェーズ 3-2)")
        self.setGeometry(100, 100, 800, 600)

        # --- Model (データを保持する場所) ---
        self.audio_data = None
        self.sample_rate = None
        self.duration = 0.0
        self.nyquist = 0.0

        # [Phase 3-2] STFTピラミッド用のパラメータとデータストレージ
        # (しきい値(秒), hop_length, n_fft)
        self.stft_level_params = [
            {"threshold": 5.0, "hop": 512, "n_fft": 2048},  # レベル0: 5秒より大きい
            {"threshold": 1.0, "hop": 256, "n_fft": 1024},  # レベル1: 1秒～5秒
            {"threshold": 0.0, "hop": 128, "n_fft": 512},  # レベル2: 1秒未満
        ]
        self.stft_levels_data = []  # 計算済みのSTFTデータ (S_db) を保持
        self.current_stft_level = -1  # 現在表示中のレベル

        # [Phase 3-2] ズーム/パン操作のデバウンス用タイマー
        self.zoom_timer = QTimer(self)
        self.zoom_timer.setSingleShot(True)  # 1回だけトリガー
        self.zoom_timer.setInterval(150)  # 150ms 操作がなければ実行
        self.zoom_timer.timeout.connect(self.update_spectrogram_view)

        # --- View (UIのセットアップ) ---
        self.init_ui()

    def init_ui(self):
        """
        UIの初期化を行います
        """
        # (メニューバー、ボタンレイアウトは変更なし)
        # ... (省略) ...
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&ファイル")
        open_action = QAction("&音声ファイルを開く...", self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
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

        # --- 波形 (Waveform) ペイン ---
        self.plot_widget = pg.PlotWidget()
        main_layout.addWidget(self.plot_widget, stretch=1)
        self.plot_widget.setBackground("w")
        self.plot_widget.setLabel("left", "振幅")
        self.plot_widget.setLabel("bottom", "時間 (秒)")
        self.plot_widget.showGrid(x=True, y=True)

        # --- スペクトログラム (Spectrogram) ペイン ---
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

        # --- マウス操作と軸連動の設定 ---
        self.plot_widget.setMouseEnabled(x=True, y=False)
        self.view_box = self.plot_widget.getPlotItem().getViewBox()
        self.view_box.setMouseMode(pg.ViewBox.PanMode)

        self.spec_widget.setMouseEnabled(x=True, y=False)  # Y軸操作は無効
        self.spec_view_box = self.spec_widget.getPlotItem().getViewBox()
        self.spec_view_box.setMouseMode(pg.ViewBox.PanMode)

        self.view_box.setXLink(self.spec_view_box)

        # [Phase 3-2] ズーム/パン操作をデバウンス用タイマーに接続
        self.view_box.sigRangeChanged.connect(self.on_range_changed)

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
        file_path, _ = QFileDialog.getOpenFileName(self, "音声ファイルを開く", "", "音声ファイル (*.wav *.mp3 *.flac)")

        if file_path:
            print(f"ファイルが選択されました: {file_path}")

            try:
                self.stop_audio()
                self.current_stft_level = -1  # リセット

                self.audio_data, self.sample_rate = librosa.load(file_path, sr=None)

                # [Phase 3-2] Modelの共通プロパティを設定
                self.duration = len(self.audio_data) / self.sample_rate
                self.nyquist = self.sample_rate / 2.0
                print(f"読み込み完了: {self.duration:.2f} 秒, SR: {self.sample_rate} Hz")

                # [Phase 3-2] スペクトログラムのY軸を固定
                self.spec_widget.setYRange(0, self.nyquist)

                # [Phase 3-2] STFTピラミッドを事前計算
                self.calculate_stft_pyramid()

                # 波形を描画 (これによりautoRangeされ、sigRangeChangedが発行される)
                self.plot_waveform()

                # [Phase 3-2] 最初の表示（レベル0）を強制的に行う
                self.update_spectrogram_view(force_update=True)

                self.play_button.setEnabled(True)
                self.stop_button.setEnabled(True)

            except Exception as e:
                print(f"ファイルの読み込みエラー: {e}")
                self.play_button.setEnabled(False)
                self.stop_button.setEnabled(False)

    def plot_waveform(self):
        """
        波形を描画する (STFT計算は分離)
        """
        if self.audio_data is None:
            return

        self.plot_widget.clear()
        time_axis = np.linspace(0, self.duration, num=len(self.audio_data))

        data_to_plot = self.audio_data
        if self.audio_data.ndim > 1:
            data_to_plot = self.audio_data[0]

        self.plot_widget.plot(time_axis, data_to_plot, pen=pg.mkPen(color=(0, 0, 200), width=1))
        print("波形を描画しました。")

        # X軸のズーム範囲を制限
        self.view_box.setLimits(xMin=0, xMax=self.duration, yMin=None, yMax=None)
        self.spec_view_box.setLimits(xMin=0, xMax=self.duration, yMin=None, yMax=self.nyquist)

        self.plot_widget.autoRange()

    # [Phase 3-2] 新しいメソッド: STFTピラミッドの事前計算
    def calculate_stft_pyramid(self):
        """
        全レベルのSTFTを事前に計算し、メモリに保持する
        """
        print("STFTピラミッドを計算中...")
        self.stft_levels_data = []  # 既存のデータをクリア

        data_to_calc = self.audio_data
        if self.audio_data.ndim > 1:
            data_to_calc = self.audio_data[0]

        for i, params in enumerate(self.stft_level_params):
            print(f"  レベル {i} (hop={params['hop']}, n_fft={params['n_fft']}) を計算中...")
            D = np.abs(librosa.stft(data_to_calc, n_fft=params["n_fft"], hop_length=params["hop"]))
            S_db = librosa.amplitude_to_db(D, ref=np.max)
            self.stft_levels_data.append(S_db)

        print("STFTピラミッドの計算が完了しました。")

    # [Phase 3-2] 新しいメソッド: ズーム/パン時のデバウンサ
    def on_range_changed(self):
        """
        ViewBoxの表示範囲が変更されたときに呼び出される
        タイマーをリスタート（デバウンス）する
        """
        if self.zoom_timer.isActive():
            self.zoom_timer.stop()
        self.zoom_timer.start()

    # [Phase 3-2] 変更されたメソッド: スペクトログラムのレイヤー切替
    def update_spectrogram_view(self, force_update=False):
        """
        現在のズームレベルに基づいて、表示するSTFTレイヤーを切り替える
        """
        if not self.stft_levels_data:
            return  # データがまだない

        # 1. 現在の表示時間幅を取得
        x_min, x_max = self.view_box.viewRange()[0]
        visible_duration = x_max - x_min

        # 2. 表示時間幅に最適なレベルを決定
        target_level_index = 0  # デフォルトはレベル0 (粗い)
        for i, params in enumerate(self.stft_level_params):
            if visible_duration <= params["threshold"]:
                target_level_index = i
            else:
                # このしきい値より大きい場合は、前のレベル(i-1)が正解
                # ただし i=0 の場合は target_level_index=0 のまま
                if i > 0:
                    target_level_index = i - 1
                break  # マッチしたらループを抜ける

        # 3. 変更がなければ何もしない (パフォーマンスのため)
        if not force_update and target_level_index == self.current_stft_level:
            return

        print(f"表示を STFT レベル {target_level_index} に切り替えます (表示時間: {visible_duration:.2f}秒)")

        # 4. データを切り替えて描画
        self.current_stft_level = target_level_index
        S_db = self.stft_levels_data[target_level_index]

        # pyqtgraphのImageItemは[X, Y]の順 (時間, 周波数) を期待する
        self.spec_image.setImage(S_db.T)

        # 画像の位置とスケールを現実の単位に合わせる
        # どのSTFTレイヤーも、全体では (0, 0) から (duration, nyquist) に
        # マッピングされることに変わりはない。
        self.spec_image.setRect(0, 0, self.duration, self.nyquist)

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
