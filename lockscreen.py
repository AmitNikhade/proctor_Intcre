
import threading
import sys
import subprocess
from disable_enable_functions import disable, enable
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QLabel, QTextEdit
from PyQt5.QtCore import Qt, QProcess, QTimer, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
from PyQt5.Qsci import QsciScintilla, QsciLexerPython
import elevate

# Elevate privileges
elevate.elevate()

class FullScreenWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set window properties
        self.setWindowTitle("Kiosk App")  # Change the window title
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.WindowDoesNotAcceptFocus)
        self.setFixedSize(1920, 1080)  # Set the size to fullscreen
        
        # Create a central widget and layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create a layout for the left side (editor and output)
        left_layout = QVBoxLayout()
        
        # Create and add the code editor
        self.code_editor = QsciScintilla(self)
        self.setup_editor()
        left_layout.addWidget(self.code_editor)
        
        # Create and add the output display
        self.output_display = QTextEdit(self)
        self.output_display.setReadOnly(True)
        left_layout.addWidget(self.output_display)
        left_layout.setStretch(0, 2)  # Set the editor to take 2/3 of the space
        left_layout.setStretch(1, 1)  # Set the output to take 1/3 of the space
        
        main_layout.addLayout(left_layout)
        
        # Create a layout for the right side (camera and buttons)
        right_layout = QVBoxLayout()
        
        # Create and add the run button
        self.run_button = QPushButton("Run Code", self)
        self.run_button.clicked.connect(self.run_code)
        right_layout.addWidget(self.run_button)
        
        # Add the camera display widget
        self.camera_label = QLabel(self)
        right_layout.addWidget(self.camera_label)

        main_layout.addLayout(right_layout)

        # Set up the camera
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update the frame every 30 ms
        
    def setup_editor(self):
        # Set up the lexer for syntax highlighting
        lexer = QsciLexerPython()
        self.code_editor.setLexer(lexer)
        
        # Set editor properties
        self.code_editor.setUtf8(True)  # Ensure Unicode support
        self.code_editor.setMarginsFont(QFont("Consolas", 12))  # Set font for line numbers
        self.code_editor.setMarginLineNumbers(1, True)  # Show line numbers
        self.code_editor.setMarginsBackgroundColor(QColor("#2e2e2e"))  # Set background color for line numbers area
        self.code_editor.setMarginWidth(1, 50)  # Set width of line numbers area
        self.code_editor.setBraceMatching(QsciScintilla.SloppyBraceMatch)  # Enable brace matching
        self.code_editor.setCaretLineVisible(True)  # Show a line under the cursor
        self.code_editor.setCaretLineBackgroundColor(QColor("#393939"))  # Set color for the line under the cursor
        
        # Set color scheme for syntax highlighting
        self.code_editor.setPaper(QColor("#1e1e1e"))  # Set editor background color
        lexer.setDefaultColor(QColor("#f8f8f2"))  # Set default text color
        lexer.setColor(QColor("#f8f8f2"), QsciLexerPython.Default)  # Set default color for Python code
        lexer.setColor(QColor("#66d9ef"), QsciLexerPython.Keyword)  # Set color for Python keywords
        lexer.setColor(QColor("#75715e"), QsciLexerPython.Comment)  # Set color for comments
        lexer.setColor(QColor("#e6db74"), QsciLexerPython.DoubleQuotedString)  # Set color for double quoted strings
        lexer.setColor(QColor("#e6db74"), QsciLexerPython.SingleQuotedString)  # Set color for single quoted strings
        lexer.setColor(QColor("#ae81ff"), QsciLexerPython.Number)  # Set color for numbers
        lexer.setColor(QColor("#ae81ff"), QsciLexerPython.TripleSingleQuotedString)  # Set color for triple single quoted strings
        lexer.setColor(QColor("#ae81ff"), QsciLexerPython.TripleDoubleQuotedString)  # Set color for triple double quoted strings
        lexer.setColor(QColor("#f8f8f2"), QsciLexerPython.ClassName)  # Set color for class names
        lexer.setColor(QColor("#f8f8f2"), QsciLexerPython.FunctionMethodName)  # Set color for function/method names
        lexer.setColor(QColor("#f8f8f2"), QsciLexerPython.Operator)  # Set color for operators
        lexer.setColor(QColor("#f8f8f2"), QsciLexerPython.Identifier)  # Set color for identifiers
        lexer.setColor(QColor("#75715e"), QsciLexerPython.CommentBlock)  # Set color for comment blocks
        lexer.setColor(QColor("#e6db74"), QsciLexerPython.UnclosedString)  # Set color for unclosed strings

    def run_code(self):
        # Get the code from the editor
        code = self.code_editor.text()
        
        # Run the code in a separate process
        process = QProcess(self)
        process.start("python", ["-c", code])
        process.waitForFinished()
        
        # Get the output and error messages
        output = process.readAllStandardOutput().data().decode()
        error = process.readAllStandardError().data().decode()
        
        # Display the output and error messages
        self.output_display.setText(output + error)

    @pyqtSlot()
    def update_frame(self):
        # Read frame from the camera
        ret, frame = self.cap.read()
        
        # If frame is read successfully, display it
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(image))

    def closeEvent(self, event):
        # Release the camera resources
        self.cap.release()
        event.ignore()  # Ignore any attempts to close the window

    def keyPressEvent(self, event):
        # Check if Ctrl+P is pressed
        if event.key() == Qt.Key_P and event.modifiers() & Qt.ControlModifier:
            enable()
            QApplication.quit()  # Exit the application

def start_app():
    app = QApplication(sys.argv)
    window = FullScreenWindow()
    
    window.showFullScreen()  # Show the window in fullscreen mode
    
    sys.exit(app.exec_())  
    
    
    
if __name__ == "__main__":

    thread1 = threading.Thread(target=start_app(), name='Thread 1')
    thread2 = threading.Thread(target=disable(), name='Thread 2')

    # Start the threads
    thread1.start()
    thread2.start()

    # Wait for both threads to finish
    thread1.join()
    thread2.join()
