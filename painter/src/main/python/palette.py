"""
Copyright (C) 2021 Abraham George Smith

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

"""
Palette: Provides a way to add, edit and remove brushes / colours / classes
"""
import random
from functools import partial

from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtGui


class BrushEditWidget(QtWidgets.QWidget):
    """
    Provide a way for a user to edit the name and colour of a brush
    """
    
    changed = QtCore.pyqtSignal()
    removed = QtCore.pyqtSignal()

    def __init__(self, name, rgba):
        super().__init__()
        self.name = name
        r, g, b, a = rgba
        self.color = QtGui.QColor(r, g, b, a) # 0-255
        self.initUI()

    def initUI(self):
        # Provide user with a way to edit the brush name
        self.layout = QtWidgets.QHBoxLayout()
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setText(self.name)
        self.name_edit.textChanged.connect(self.text_changed)
        self.layout.addWidget(self.name_edit)

        self.color_btn = QtWidgets.QPushButton(' ')
        self.color_btn.setStyleSheet(f"background-color:{self.color.name()};")
        self.color_btn.clicked.connect(self.color_btn_clicked)
        self.layout.addWidget(self.color_btn)

        self.remove_btn = QtWidgets.QPushButton('Remove')
        self.remove_btn.clicked.connect(self.removed.emit)
        self.layout.addWidget(self.remove_btn)

        self.setLayout(self.layout)


    def color_btn_clicked(self):
        # When the user clicks the color label. Let them pick a new color
        show_alpha_option = QtWidgets.QColorDialog.ColorDialogOption(1)
        new_color = QtWidgets.QColorDialog.getColor(
            self.color,
            options=show_alpha_option)

        if new_color.isValid():
            self.color = new_color
            self.color_btn.setStyleSheet(f"background-color:{self.color.name()};")
            self.changed.emit()

    def text_changed(self):
        new_text = self.name_edit.text()
        self.name = new_text
        self.changed.emit()


def get_random_rgba():
    r = 255 * random.random()
    g = 255 * random.random()
    b = 255 * random.random()
    a = 255
    return [r, g, b, a]


class PaletteEditWidget(QtWidgets.QWidget):
    """ Add, edit and remove brushes """

    changed = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        label = QtWidgets.QLabel()
        label.setText("Palette: Edit your brushes")
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(label)

        # Use a container for the brushes so the add brush widget can go after.
        self.brushes_container = QtWidgets.QWidget()
        self.brushes_layout = QtWidgets.QVBoxLayout()
        self.brushes_container.setLayout(self.brushes_layout)
        self.layout.addWidget(self.brushes_container)

        
        # These are the default brushes
        # name, colour (r,g,b,a), keyboard shortcut
        # The pythonpanic land cover classes.
        default_brushes = [
            ('forest', (255, 0, 0, 180), '1'),
            ('wetland', (255, 0, 255, 180), '2'),
            ('heather', (255, 200, 0, 180), '3'),
            ('water', (0, 0, 255, 180), '4'),
                    ]
        
        self.brush_widgets = []
        for name, rgba, _ in default_brushes:
            self.add_brush(name, rgba)

        self.add_brush_btn = QtWidgets.QPushButton('Add brush')
        self.add_brush_btn.clicked.connect(self.add_brush)
        self.layout.addWidget(self.add_brush_btn)

    def get_new_name(self):
        return f"Brush {len(self.brush_widgets)}"


    def add_brush(self, name=None, rgba=None):
        if not name:
            name = self.get_new_name()
        if not rgba:
            rgba = get_random_rgba()

        brush = BrushEditWidget(name, rgba)
        self.brush_widgets.append(brush)

        brush.removed.connect(self.remove_brush)
        self.brushes_layout.addWidget(brush)
        self.changed.emit()

    def remove_brush(self):
        brush = self.sender()
        self.brush_widgets.remove(brush)
        self.brushes_layout.removeWidget(brush)
        self.changed.emit()


    def get_brush_data(self):
        """ Used for saving the brush data to JSON file """
       
        # Background cannot be edited or removed
        brush_data = [
            ('Background', (0, 255, 0, 180), 'W'),
        ]

        for brush_widget in self.brush_widgets:
            # name, rgba, keyboard shortcut
            # Alpha is fixed: a transparent brush would look like the eraser
            # to the trainer.
            r, g, b, _ = brush_widget.color.getRgb()
            brush_data.append([brush_widget.name,
                               (r, g, b, 180),
                               str(len(brush_data))])
        return brush_data

    def get_validation_error(self):
        """ Message explaining why the brushes can't be used, or None """
        brush_data = self.get_brush_data()
        if len(brush_data) < 2:
            return 'At least one foreground class must be specified'
        if len(brush_data) > 10:
            return 'At most 9 classes are supported (keyboard shortcuts 1-9)'
        names = [b[0].strip() for b in brush_data]
        if '' in names or len(set(names)) != len(names):
            return 'Each brush must have a unique name'
        colours = [tuple(b[1][:3]) for b in brush_data]
        if len(set(colours)) != len(colours):
            return 'Each brush must have a different colour'
        return None
