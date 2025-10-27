import sys
import numpy as np
import librosa
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout, QWidget, QMenuBar
from PyQt6.QtGui import QAction


class PySurferWindow(QMainWindow):
    """
    メインウィンドウクラス
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PySurfer (Phase 1)")
        self.setGeometry(100, 100, 800, 400)  # x, y, width, height

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
        file_menu = menu_bar.addMenu("&File")

        # "File" メニューに "Open" アクションを追加
        open_action = QAction("&Open Audio File...", self)
        open_action.triggered.connect(self.open_file)  # クリック時の動作を接続
        file_menu.addAction(open_action)

        # メインとなるウィジェット
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # レイアウトの作成 (垂直にウィジェットを並べる)
        layout = QVBoxLayout(central_widget)

        # PyQtGraphのプロットウィジェットを作成
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)  # レイアウトに追加

        # プロットウィジェットの初期設定
        self.plot_widget.setBackground("w")  # 背景色を白に
        self.plot_widget.setLabel("left", "Amplitude")
        self.plot_widget.setLabel("bottom", "Time (s)")
        self.plot_widget.showGrid(x=True, y=True)

    # --- Controller (イベント処理) ---
    def open_file(self):
        """
        "Open" アクションが実行されたときの処理
        """
        print("Opening file dialog...")

        # 1. ファイル選択ダイアログを表示
        # QFileDialog.getOpenFileName(parent, caption, directory, filter)
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "", "Audio Files (*.wav *.mp3 *.flac)"  # 初期ディレクトリ  # ファイルフィルタ
        )

        if file_path:  # ファイルが選択された場合
            print(f"File selected: {file_path}")

            # 2. Librosaで音声ファイルを読み込む (Modelの更新)
            try:
                # sr=None で元のサンプリングレートを維持して読み込む
                self.audio_data, self.sample_rate = librosa.load(file_path, sr=None)
                print(f"Loaded audio: {len(self.audio_data)} samples, SR: {self.sample_rate} Hz")

                # 3. 読み込んだデータをプロットする (Viewの更新)
                self.plot_waveform()

            except Exception as e:
                print(f"Error loading file: {e}")
                # (将来的にここにエラーダイアログを出す)

    def plot_waveform(self):
        """
        Modelのデータ (self.audio_data) を View (plot_widget) に描画する
        """
        if self.audio_data is None:
            return

        # 4. 波形データをPyQtGraphで表示

        # まずプロットをクリア
        self.plot_widget.clear()

        # 横軸（時間）のデータを作成
        # np.arange(N) / sr
        duration = len(self.audio_data) / self.sample_rate
        time_axis = np.linspace(0, duration, num=len(self.audio_data))

        # チャンネル処理 (モノラル/ステレオ)
        if self.audio_data.ndim > 1:
            # ステレオや多チャンネルの場合、ここでは最初のチャンネルのみ表示
            data_to_plot = self.audio_data[0]
            print("Stereo file detected, plotting first channel.")
        else:
            # モノラルの場合
            data_to_plot = self.audio_data

        # データをプロット
        self.plot_widget.plot(time_axis, data_to_plot, pen=pg.mkPen(color=(0, 0, 200), width=1))  # ペンの色 (青)

        # 表示範囲をリセット
        self.plot_widget.autoRange()
        print("Waveform plotted.")


# --- アプリケーションの実行 ---
if __name__ == "__main__":
    # PyQtGraphのアンチエイリアスを有効化 (線が滑らかになる)
    pg.setConfigOptions(antialias=True)

    app = QApplication(sys.argv)
    window = PySurferWindow()
    window.show()
    sys.exit(app.exec())
