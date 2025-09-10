from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QLineEdit, QLabel
import sys
import pickle
from culler import cull_photos

class PhotoCullerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Photo Culler")
        self.setGeometry(100, 100, 400, 200)
        self.model = self.load_model()
        self.init_ui()

    def load_model(self):
        with open("model/trained_model.pkl", "rb") as f:
            return pickle.load(f)

    def init_ui(self):
        # Source folder
        self.src_button = QPushButton("Select Source Folder", self)
        self.src_button.move(50, 30)
        self.src_button.clicked.connect(self.select_source)

        # Destination folder
        self.dst_button = QPushButton("Select Destination Folder", self)
        self.dst_button.move(50, 70)
        self.dst_button.clicked.connect(self.select_dest)

        # Threshold input
        self.threshold_label = QLabel("Threshold:", self)
        self.threshold_label.move(50, 110)
        self.threshold_input = QLineEdit(self)
        self.threshold_input.move(150, 110)
        self.threshold_input.setText("0.6")

        # Run button
        self.run_button = QPushButton("Run Culler", self)
        self.run_button.move(50, 150)
        self.run_button.clicked.connect(self.run_culler)

    def select_source(self):
        self.source_folder = QFileDialog.getExistingDirectory(self, "Select Source Folder")

    def select_dest(self):
        self.dest_folder = QFileDialog.getExistingDirectory(self, "Select Destination Folder")

    def run_culler(self):
        threshold = float(self.threshold_input.text())
        cull_photos(self.model, self.source_folder, self.dest_folder, threshold)
        print("Culling completed!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PhotoCullerApp()
    window.show()
    sys.exit(app.exec_())
