import sys
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from modules.image_generation.sd_workflow_tab import SdWorkflowWidget
from utils.gui_entry import configure_qt_application_attributes


if __name__ == "__main__":
    configure_qt_application_attributes(QApplication, Qt)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = SdWorkflowWidget()
    window.show()
    sys.exit(app.exec())
