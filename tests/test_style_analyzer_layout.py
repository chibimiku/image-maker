import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtCore import QPoint
from PyQt6.QtWidgets import QApplication
from modules.image_analysis.style_analyzer import StyleAnalyzerWidget


class StyleAnalyzerLayoutTests(unittest.TestCase):
    def test_start_cancel_and_status_stay_visible_when_options_scroll(self):
        app = QApplication.instance() or QApplication([])
        widget = StyleAnalyzerWidget(lambda: ("", "", ""))
        self.assertEqual(widget.total_rounds_spin.value(), 3)
        self.assertEqual(widget.images_per_round_spin.value(), 4)
        widget.resize(1100, 650)
        widget.show()
        app.processEvents()
        try:
            controls = widget.controls_scroll.widget()
            self.assertFalse(controls.isAncestorOf(widget.analyze_btn))
            self.assertFalse(controls.isAncestorOf(widget.cancel_btn))
            self.assertFalse(controls.isAncestorOf(widget.status_label))
            initial = widget.analyze_btn.mapTo(widget, QPoint(0, 0))
            for value in (0, widget.controls_scroll.verticalScrollBar().maximum()):
                widget.controls_scroll.verticalScrollBar().setValue(value)
                app.processEvents()
                self.assertEqual(widget.analyze_btn.mapTo(widget, QPoint(0, 0)), initial)
                for button in (widget.analyze_btn, widget.cancel_btn):
                    self.assertTrue(widget.rect().contains(button.mapTo(widget, button.rect().center())))
                    self.assertTrue(button.isVisible())
            self.assertGreaterEqual(widget.result_tabs.height(), 150)
        finally:
            widget.close()


if __name__ == "__main__":
    unittest.main()
