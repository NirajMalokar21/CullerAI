from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QFileDialog, QLineEdit, QLabel,
    QVBoxLayout, QHBoxLayout, QProgressBar, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import sys
import pickle
import os
from culler import cull_photos

def load_model():
    """Load the trained model in a PyInstaller-friendly way."""
    if getattr(sys, "frozen", False):
        # PyInstaller bundle
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "model", "trained_model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)



class CullerThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, model, src, dst, threshold):
        super().__init__()
        self.model = model
        self.src = src
        self.dst = dst
        self.threshold = threshold

    def run(self):
        # Wrap culling in a generator so we can emit progress
        import os
        from culler import score_image
        import shutil

        files = [f for f in os.listdir(self.src)]
        total = len(files)
        processed = 0

        for f in files:
            processed += 1
            full_path = os.path.join(self.src, f)
            try:
                score = score_image(self.model, full_path)
                if score >= self.threshold:
                    shutil.copy(full_path, self.dst)
            except Exception as e:
                print("Skipping", f, "error:", e)

            self.progress.emit(int((processed / total) * 100))

        self.finished.emit()


class PhotoCullerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Photo Culler")
        self.setGeometry(300, 200, 500, 250)
        self.model = load_model()
        self.source_folder = ""
        self.dest_folder = ""
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Source folder
        src_layout = QHBoxLayout()
        self.src_button = QPushButton("Select Source Folder")
        self.src_button.clicked.connect(self.select_source)
        src_layout.addWidget(self.src_button)
        layout.addLayout(src_layout)

        # Destination folder
        dst_layout = QHBoxLayout()
        self.dst_button = QPushButton("Select Destination Folder")
        self.dst_button.clicked.connect(self.select_dest)
        dst_layout.addWidget(self.dst_button)
        layout.addLayout(dst_layout)

        # Threshold input
        thresh_layout = QHBoxLayout()
        self.threshold_label = QLabel("Threshold:")
        self.threshold_input = QLineEdit()
        self.threshold_input.setText("1.0")
        thresh_layout.addWidget(self.threshold_label)
        thresh_layout.addWidget(self.threshold_input)
        layout.addLayout(thresh_layout)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress)

        # Run button
        self.run_button = QPushButton("Run Culler")
        self.run_button.clicked.connect(self.run_culler)
        layout.addWidget(self.run_button)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Apply layout
        self.setLayout(layout)

        # Apply a modern style
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f2f5;
                font-family: Arial;
                font-size: 14px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QLineEdit {
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
            QLabel {
                font-weight: bold;
            }
            QProgressBar {
                border: 1px solid #bbb;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 20px;
            }
        """)

    def select_source(self):
        self.source_folder = QFileDialog.getExistingDirectory(self, "Select Source Folder")
        self.status_label.setText(f"Source: {self.source_folder}")

    def select_dest(self):
        self.dest_folder = QFileDialog.getExistingDirectory(self, "Select Destination Folder")
        self.status_label.setText(f"Destination: {self.dest_folder}")

    def run_culler(self):
        if not self.source_folder or not self.dest_folder:
            QMessageBox.warning(self, "Error", "Please select both source and destination folders.")
            return

        threshold = float(self.threshold_input.text())

        # Run in background thread
        self.thread = CullerThread(self.model, self.source_folder, self.dest_folder, threshold)
        self.thread.progress.connect(self.progress.setValue)
        self.thread.finished.connect(self.on_finished)
        self.thread.start()

        self.status_label.setText("Culling in progress...")

    def on_finished(self):
        self.status_label.setText("✅ Culling completed!")
        QMessageBox.information(self, "Done", "Culling completed successfully!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PhotoCullerApp()
    window.show()
    sys.exit(app.exec_())
