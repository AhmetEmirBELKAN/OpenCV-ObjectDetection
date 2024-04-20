import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow
from design import Ui_MainWindow  # 'your_converted_file' adını çevirdiğiniz dosyanın adıyla değiştirin

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())