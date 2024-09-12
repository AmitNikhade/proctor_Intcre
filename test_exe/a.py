from PyQt5.QtWidgets import QLabel, QDialog, QVBoxLayout, QApplication
from PyQt5.QtGui import QMovie
import sys
import time
from PyQt5.QtCore import Qt

class LoadingScreen(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setStyleSheet("background-color: #1e1e1e; border-radius: 10px;")
        
        # Set up the layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create a label to display the animated GIF
        self.loading_label = QLabel(self)
        layout.addWidget(self.loading_label)
        
        # Load the animated GIF
        self.movie = QMovie("test_exe\gif.mp4")  # Ensure you have a loading.gif in your directory
        self.loading_label.setMovie(self.movie)
        self.movie.start()

        # Set the size of the loading dialog
        self.setFixedSize(200, 200)
        
    def close_loading(self):
        self.movie.stop()
        self.close()

def start_loading_screen():
    app = QApplication(sys.argv)
    
    # Create and show the loading screen
    loading_screen = LoadingScreen()
    loading_screen.show()

    # Simulate a long-running task (you should replace this with your actual task)
    time.sleep(3)  # This would be where your actual loading happens

    # After the task completes, close the loading screen
    loading_screen.close_loading()

    sys.exit(app.exec_())

if __name__ == "__main__":
    start_loading_screen()
