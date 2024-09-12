import sys
import http.client
import json
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QPushButton, QComboBox

class CodeEditor(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()

        # Language selection dropdown
        lang_layout = QHBoxLayout()
        self.lang_label = QLabel("Select Language:")
        self.lang_combo = QComboBox()
        self.fetch_languages()
        
        lang_layout.addWidget(self.lang_label)
        lang_layout.addWidget(self.lang_combo)
        layout.addLayout(lang_layout)
        
        # Code editor
        self.code_editor = QTextEdit()
        layout.addWidget(self.code_editor)
        
        # Run button
        self.run_button = QPushButton("Run Code")
        self.run_button.clicked.connect(self.run_code)
        layout.addWidget(self.run_button)
        
        # Output display
        self.output_display = QTextEdit()
        self.output_display.setReadOnly(True)
        layout.addWidget(self.output_display)
        
        self.setLayout(layout)
        self.setWindowTitle('PyQt Code Editor with Judge0 API')
        self.resize(600, 400)

    def fetch_languages(self):
        conn = http.client.HTTPSConnection("judge0-extra-ce.p.rapidapi.com")
        headers = {
            'x-rapidapi-key': "0401bc71b7msh375aa9a2536aceep11a122jsnec82293642a7",
            'x-rapidapi-host': "judge0-extra-ce.p.rapidapi.com"
        }
        conn.request("GET", "/languages", headers=headers)
        res = conn.getresponse()
        data = res.read().decode("utf-8")
        languages = json.loads(data)
        
        for lang in languages:
            self.lang_combo.addItem(lang['name'], lang['id'])

        # Manually add JavaScript if not present
        js_present = any(self.lang_combo.itemText(i).lower() == 'javascript' for i in range(self.lang_combo.count()))
        if not js_present:
            self.lang_combo.addItem("JavaScript (Node.js)", 63)  # ID 63 is typically used for JavaScript in Judge0

    def run_code(self):
        source_code = self.code_editor.toPlainText()
        language_id = self.lang_combo.currentData()

        if not source_code.strip():
            self.output_display.setText("Please enter code to run.")
            return

        conn = http.client.HTTPSConnection("judge0-extra-ce.p.rapidapi.com")
        headers = {
            'x-rapidapi-key': "0401bc71b7msh375aa9a2536aceep11a122jsnec82293642a7",
            'x-rapidapi-host': "judge0-extra-ce.p.rapidapi.com",
            'Content-Type': 'application/json'
        }

        payload = json.dumps({
            "language_id": language_id,
            "source_code": source_code
        })

        conn.request("POST", "/submissions?base64_encoded=false&wait=true", payload, headers)
        res = conn.getresponse()
        data = res.read().decode("utf-8")
        result = json.loads(data)

        if 'stdout' in result and result['stdout']:
            self.output_display.setText(result['stdout'])
        elif 'stderr' in result and result['stderr']:
            self.output_display.setText(f"Error: {result['stderr']}")
        elif 'compile_output' in result and result['compile_output']:
            self.output_display.setText(f"Compilation Error: {result['compile_output']}")
        else:
            self.output_display.setText(f"Error occurred: {json.dumps(result, indent=4)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    editor = CodeEditor()
    editor.show()
    sys.exit(app.exec_())
