"""
Tab 1: Animated Plasma Heatmap (Sine Interference) with extended controls
Tab 2: Neural Network where weights = live pixel values from Tab 1
"""

import sys
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QSlider, QPushButton,
                               QComboBox, QGroupBox, QFormLayout, QTextEdit, QTabWidget,
                               QScrollArea, QCheckBox)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QTimer


# ========== Heatmap Colormaps ==========
def make_jet_colormap():
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        r = int(np.clip(1.5 - np.abs(4 * t - 3), 0, 1) * 255)
        g = int(np.clip(1.5 - np.abs(4 * t - 2), 0, 1) * 255)
        b = int(np.clip(1.5 - np.abs(4 * t - 1), 0, 1) * 255)
        lut[i] = [r, g, b]
    return lut

def make_viridis_colormap():
    stops = [(0.0, (68, 1, 84)), (0.25, (59, 82, 139)), (0.5, (33, 145, 140)),
             (0.75, (94, 201, 98)), (1.0, (253, 231, 37))]
    return _interpolate_stops(stops)

def make_plasma_colormap():
    stops = [(0.0, (13, 8, 135)), (0.25, (126, 3, 168)), (0.5, (204, 71, 120)),
             (0.75, (248, 149, 64)), (1.0, (240, 249, 33))]
    return _interpolate_stops(stops)

def make_hot_colormap():
    stops = [(0.0, (0, 0, 0)), (0.33, (255, 0, 0)), (0.66, (255, 255, 0)), (1.0, (255, 255, 255))]
    return _interpolate_stops(stops)

def make_cool_colormap():
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        lut[i] = [int(t * 255), int((1 - t) * 255), 255]
    return lut

def make_grayscale_colormap():
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        lut[i] = [i, i, i]
    return lut

def make_inferno_colormap():
    stops = [(0.0, (0, 0, 4)), (0.25, (87, 16, 110)), (0.5, (188, 55, 84)),
             (0.75, (249, 142, 9)), (1.0, (252, 255, 164))]
    return _interpolate_stops(stops)

def _interpolate_stops(stops):
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
            if t0 <= t <= t1:
                a = (t - t0) / (t1 - t0)
                lut[i] = [int(c0[j] + (c1[j] - c0[j]) * a) for j in range(3)]
                break
    return lut

COLORMAPS = {
    "Jet": make_jet_colormap(),
    "Viridis": make_viridis_colormap(),
    "Plasma": make_plasma_colormap(),
    "Hot": make_hot_colormap(),
    "Cool": make_cool_colormap(),
    "Grayscale": make_grayscale_colormap(),
    "Inferno": make_inferno_colormap(),
}


# ========== Plasma Simulation ==========
class PlasmaSim:
    """Generates an animated 2D plasma effect using sine wave interference."""
    
    def __init__(self, width=32, height=32):
        self.w, self.h = width, height
        
        # Precompute coordinate grids
        self.y, self.x = np.ogrid[:self.h, :self.w]
        self.xn = self.x / self.w * 2 * np.pi
        self.yn = self.y / self.h * 2 * np.pi
        
        # Controllable parameters
        self.time = 0.0
        self.freq = 3.0
        self.wave_count = 4
        self.speed_mult = 1.0
        self.turbulence = 0.0
        self.center_x = self.w / 2.0
        self.center_y = self.h / 2.0
        self.blend_mode = "additive"  # additive, multiplicative, max
        self.manual_phase = 0.0
        self.invert = False
        self.colormap_name = "Jet"
        
        # Precompute turbulence noise (static per reset)
        self._regen_noise()
        
    def _regen_noise(self, seed=None):
        rng = np.random.default_rng(seed)
        self.noise = rng.standard_normal((self.h, self.w)).astype(np.float32) * 0.5
        
    def reset(self, new_seed=True):
        self.time = 0.0
        if new_seed:
            self._regen_noise()
        
    def step(self, dt=0.1):
        self.time += dt * self.speed_mult
        
    def get_pixels_2d(self):
        """Get current state as normalized 2D grid (0.0 to 1.0)."""
        f = self.freq
        t = self.time + self.manual_phase
        
        waves = []
        # Wave 1: horizontal
        waves.append(np.sin(self.xn * f + t * 1.0))
        # Wave 2: vertical
        waves.append(np.sin(self.yn * f + t * 1.3))
        # Wave 3: diagonal
        waves.append(np.sin((self.xn + self.yn) * f * 0.7 + t * 0.9))
        # Wave 4: radial (from configurable center)
        dist = np.sqrt((self.x - self.center_x)**2 + (self.y - self.center_y)**2)
        waves.append(np.sin(dist * f * 0.5 + t * 1.1))
        # Wave 5: anti-diagonal
        waves.append(np.sin((self.xn - self.yn) * f * 0.8 + t * 0.6))
        # Wave 6: secondary radial
        waves.append(np.cos(dist * f * 0.3 - t * 0.8))
        # Wave 7: high-freq horizontal ripple
        waves.append(np.sin(self.xn * f * 2.0 + self.yn * f * 0.5 + t * 1.5))
        # Wave 8: spiral-ish
        angle = np.arctan2(self.y - self.center_y, self.x - self.center_x)
        waves.append(np.sin(angle * 3 + dist * f * 0.2 + t * 0.7))
        
        selected = waves[:self.wave_count]
        
        # Blend modes
        if self.blend_mode == "additive":
            plasma = np.sum(selected, axis=0) / len(selected)
        elif self.blend_mode == "multiplicative":
            # Shift waves to positive range, multiply, then normalize
            pos = [(w + 1.0) / 2.0 for w in selected]
            product = pos[0]
            for p in pos[1:]:
                product = product * p
            plasma = product * 2.0 - 1.0
        elif self.blend_mode == "max":
            plasma = np.maximum.reduce(selected)
        else:
            plasma = np.sum(selected, axis=0) / len(selected)
            
        # Add turbulence
        if self.turbulence > 0:
            plasma = plasma * (1.0 - self.turbulence) + self.noise * self.turbulence
            
        # Normalize to 0..1
        result = (plasma + 1.0) / 2.0
        result = np.clip(result, 0.0, 1.0)
        
        if self.invert:
            result = 1.0 - result
            
        return result

    def get_pixels(self):
        """Get current state as normalized 1D pixel values (-1.0 to 1.0)."""
        return self.get_pixels_2d().flatten() * 2.0 - 1.0


# ========== Benchmark Datasets ==========
class BenchmarkDataset:
    @staticmethod
    def quadrant_classification(n_samples=500, size=32, seed=42):
        rng = np.random.default_rng(seed)
        X = np.zeros((n_samples, size, size), dtype=np.float32)
        y = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            q = rng.integers(0, 4)
            ranges = [(4, 12, 4, 12), (20, 28, 4, 12), (4, 12, 20, 28), (20, 28, 20, 28)]
            cx_lo, cx_hi, cy_lo, cy_hi = ranges[q]
            cx, cy = rng.integers(cx_lo, cx_hi), rng.integers(cy_lo, cy_hi)
            r = 2
            yy, xx = np.ogrid[-r:r+1, -r:r+1]
            mask = xx*xx + yy*yy <= r*r
            y0, x0 = max(0, cy-r), max(0, cx-r)
            y1, x1 = min(size, cy+r+1), min(size, cx+r+1)
            m_y0, m_x0 = y0 - (cy-r), x0 - (cx-r)
            m_y1, m_x1 = m_y0 + (y1-y0), m_x0 + (x1-x0)
            X[i, y0:y1, x0:x1][mask[m_y0:m_y1, m_x0:m_x1]] = 1.0
            y[i] = q
        return X.reshape(n_samples, -1), y, 4
    
    @staticmethod
    def dot_count(n_samples=500, size=32, seed=42):
        rng = np.random.default_rng(seed)
        X = np.zeros((n_samples, size, size), dtype=np.float32)
        y = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            n_dots = rng.integers(0, 5)
            y[i] = n_dots
            for _ in range(n_dots):
                cx, cy = rng.integers(4, size-4), rng.integers(4, size-4)
                r = 2
                yy, xx = np.ogrid[-r:r+1, -r:r+1]
                mask = xx*xx + yy*yy <= r*r
                y0, x0 = max(0, cy-r), max(0, cx-r)
                y1, x1 = min(size, cy+r+1), min(size, cx+r+1)
                m_y0, m_x0 = y0 - (cy-r), x0 - (cx-r)
                m_y1, m_x1 = m_y0 + (y1-y0), m_x0 + (x1-x0)
                X[i, y0:y1, x0:x1][mask[m_y0:m_y1, m_x0:m_x1]] = 1.0
        return X.reshape(n_samples, -1), y, 5


# ========== Tab 1: Animated Plasma Heatmap ==========
class PlasmaAnimationTab(QWidget):
    def __init__(self, sim):
        super().__init__()
        self.sim = sim
        
        root = QHBoxLayout(self)
        
        # Left: Visualization
        viz_layout = QVBoxLayout()
        self.plasma_img = QLabel()
        self.plasma_img.setFixedSize(450, 450)
        self.plasma_img.setStyleSheet("background: #000; border: 1px solid #444;")
        viz_layout.addWidget(QLabel("<b>Animated Plasma Heatmap (32×32)</b>"))
        viz_layout.addWidget(self.plasma_img)
        root.addLayout(viz_layout)
        
        # Right: Scrollable Controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(340)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        ctrl_container = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_container)
        ctrl_layout.setContentsMargins(6, 6, 6, 6)
        ctrl_layout.setSpacing(8)
        
        # --- Animation Group ---
        anim_group = QGroupBox("Animation")
        anim_form = QFormLayout(anim_group)
        
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 30)
        self.speed_slider.setValue(5)
        anim_form.addRow("Steps / frame:", self.speed_slider)
        
        self.speed_mult_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_mult_slider.setRange(1, 50)
        self.speed_mult_slider.setValue(10)
        self.speed_mult_slider.valueChanged.connect(self._update_speed_mult)
        self.speed_mult_label = QLabel("1.0x")
        self.speed_mult_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        anim_form.addRow("Wave speed:", self.speed_mult_slider)
        anim_form.addRow("", self.speed_mult_label)
        
        self.phase_slider = QSlider(Qt.Orientation.Horizontal)
        self.phase_slider.setRange(0, 1000)
        self.phase_slider.setValue(0)
        self.phase_slider.valueChanged.connect(self._update_phase)
        self.phase_label = QLabel("0.00")
        self.phase_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        anim_form.addRow("Manual phase:", self.phase_slider)
        anim_form.addRow("", self.phase_label)
        
        btn_layout = QHBoxLayout()
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setCheckable(True)
        self.reset_btn = QPushButton("Reset")
        btn_layout.addWidget(self.pause_btn)
        btn_layout.addWidget(self.reset_btn)
        anim_form.addRow(btn_layout)
        
        ctrl_layout.addWidget(anim_group)
        
        # --- Pattern Group ---
        pattern_group = QGroupBox("Pattern")
        pattern_form = QFormLayout(pattern_group)
        
        self.freq_slider = QSlider(Qt.Orientation.Horizontal)
        self.freq_slider.setRange(5, 100)
        self.freq_slider.setValue(30)
        self.freq_slider.valueChanged.connect(self._update_freq)
        self.freq_label = QLabel("3.0")
        self.freq_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        pattern_form.addRow("Frequency:", self.freq_slider)
        pattern_form.addRow("", self.freq_label)
        
        self.wave_slider = QSlider(Qt.Orientation.Horizontal)
        self.wave_slider.setRange(1, 8)
        self.wave_slider.setValue(4)
        self.wave_slider.valueChanged.connect(self._update_wave_count)
        self.wave_label = QLabel("4")
        self.wave_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        pattern_form.addRow("Wave count:", self.wave_slider)
        pattern_form.addRow("", self.wave_label)
        
        self.blend_combo = QComboBox()
        self.blend_combo.addItems(["additive", "multiplicative", "max"])
        self.blend_combo.currentTextChanged.connect(self._update_blend)
        pattern_form.addRow("Blend mode:", self.blend_combo)
        
        self.turbulence_slider = QSlider(Qt.Orientation.Horizontal)
        self.turbulence_slider.setRange(0, 100)
        self.turbulence_slider.setValue(0)
        self.turbulence_slider.valueChanged.connect(self._update_turbulence)
        self.turbulence_label = QLabel("0.00")
        self.turbulence_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        pattern_form.addRow("Turbulence:", self.turbulence_slider)
        pattern_form.addRow("", self.turbulence_label)
        
        self.regen_noise_btn = QPushButton("Re-roll Noise")
        self.regen_noise_btn.clicked.connect(lambda: self.sim._regen_noise())
        pattern_form.addRow(self.regen_noise_btn)
        
        ctrl_layout.addWidget(pattern_group)
        
        # --- Geometry Group ---
        geom_group = QGroupBox("Geometry")
        geom_form = QFormLayout(geom_group)
        
        self.cx_slider = QSlider(Qt.Orientation.Horizontal)
        self.cx_slider.setRange(0, 31)
        self.cx_slider.setValue(15)
        self.cx_slider.valueChanged.connect(self._update_center)
        geom_form.addRow("Center X:", self.cx_slider)
        
        self.cy_slider = QSlider(Qt.Orientation.Horizontal)
        self.cy_slider.setRange(0, 31)
        self.cy_slider.setValue(15)
        self.cy_slider.valueChanged.connect(self._update_center)
        geom_form.addRow("Center Y:", self.cy_slider)
        
        self.invert_check = QCheckBox("Invert colors")
        self.invert_check.stateChanged.connect(self._update_invert)
        geom_form.addRow(self.invert_check)
        
        ctrl_layout.addWidget(geom_group)
        
        # --- Appearance Group ---
        appear_group = QGroupBox("Appearance")
        appear_form = QFormLayout(appear_group)
        
        self.colormap_combo = QComboBox()
        for name in COLORMAPS.keys():
            self.colormap_combo.addItem(name)
        self.colormap_combo.currentTextChanged.connect(self._update_colormap)
        appear_form.addRow("Colormap:", self.colormap_combo)
        
        ctrl_layout.addWidget(appear_group)
        ctrl_layout.addStretch()
        
        scroll.setWidget(ctrl_container)
        root.addWidget(scroll)

    # --- Control callbacks ---
    def _update_speed_mult(self, v):
        self.sim.speed_mult = v / 10.0
        self.speed_mult_label.setText(f"{self.sim.speed_mult:.1f}x")
        
    def _update_phase(self, v):
        self.sim.manual_phase = v / 100.0
        self.phase_label.setText(f"{self.sim.manual_phase:.2f}")
        
    def _update_freq(self, v):
        self.sim.freq = v / 10.0
        self.freq_label.setText(f"{self.sim.freq:.1f}")
        
    def _update_wave_count(self, v):
        self.sim.wave_count = v
        self.wave_label.setText(str(v))
        
    def _update_blend(self, text):
        self.sim.blend_mode = text
        
    def _update_turbulence(self, v):
        self.sim.turbulence = v / 100.0
        self.turbulence_label.setText(f"{self.sim.turbulence:.2f}")
        
    def _update_center(self, _):
        self.sim.center_x = self.cx_slider.value()
        self.sim.center_y = self.cy_slider.value()
        
    def _update_invert(self, state):
        self.sim.invert = bool(state)
        
    def _update_colormap(self, name):
        self.sim.colormap_name = name

    def update_visuals(self):
        """Called by the main timer to update the heatmap."""
        img = self.sim.get_pixels_2d()
        idx = (img * 255).astype(np.uint8)
        cmap = COLORMAPS.get(self.sim.colormap_name, COLORMAPS["Jet"])
        rgb = np.ascontiguousarray(cmap[idx])
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(self.plasma_img.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.plasma_img.setPixmap(scaled)


# ========== Tab 2: NN with Plasma Weights ==========
class NNWeightsTab(QWidget):
    def __init__(self, sim):
        super().__init__()
        self.sim = sim
        self.current_sample_idx = 0
        
        self.X_test, self.y_test, self.n_classes = BenchmarkDataset.quadrant_classification(500, seed=99)
        
        root = QHBoxLayout(self)
        
        # Left: Visualizations
        viz_layout = QVBoxLayout()
        
        self.weight_img = QLabel()
        self.weight_img.setFixedSize(450, 180)
        self.weight_img.setStyleSheet("background: #000; border: 1px solid #444;")
        viz_layout.addWidget(QLabel("<b>NN Weight Heatmap (Derived from Plasma Pixels)</b>"))
        viz_layout.addWidget(self.weight_img)
        
        self.sample_img = QLabel()
        self.sample_img.setFixedSize(450, 180)
        self.sample_img.setStyleSheet("background: #000; border: 1px solid #444;")
        viz_layout.addWidget(QLabel("<b>Sample Input Heatmap (32 × 32)</b>"))
        viz_layout.addWidget(self.sample_img)
        
        root.addLayout(viz_layout)
        
        # Right: Controls & Metrics
        ctrl = QGroupBox("Neural Network Inference")
        ctrl.setFixedWidth(300)
        form = QFormLayout(ctrl)
        
        self.benchmark_combo = QComboBox()
        self.benchmark_combo.addItems(["Quadrant Classification (4 classes)", "Dot Count (5 classes)"])
        self.benchmark_combo.currentTextChanged.connect(self._change_benchmark)
        form.addRow("Benchmark:", self.benchmark_combo)
        
        self.sample_slider = QSlider(Qt.Orientation.Horizontal)
        self.sample_slider.setRange(0, 499)
        self.sample_slider.setValue(0)
        self.sample_slider.valueChanged.connect(self._change_sample)
        form.addRow("Dataset Sample:", self.sample_slider)
        
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setFixedHeight(220)
        form.addRow(self.metrics_text)
        
        root.addWidget(ctrl)
        
        self._render_sample()
        self._update_metrics()

    def _change_benchmark(self, name):
        if "Quadrant" in name:
            self.X_test, self.y_test, self.n_classes = BenchmarkDataset.quadrant_classification(500, seed=99)
        else:
            self.X_test, self.y_test, self.n_classes = BenchmarkDataset.dot_count(500, seed=99)
        self._update_metrics()

    def _change_sample(self, idx):
        self.current_sample_idx = idx
        self._render_sample()
        self._update_metrics()

    def _render_sample(self):
        sample = self.X_test[self.current_sample_idx].reshape(32, 32)
        idx_s = (sample * 255).astype(np.uint8)
        rgb_s = np.ascontiguousarray(COLORMAPS["Jet"][idx_s])
        h_s, w_s = rgb_s.shape[:2]
        qimg_s = QImage(rgb_s.data, w_s, h_s, 3 * w_s, QImage.Format_RGB888).copy()
        pix_s = QPixmap.fromImage(qimg_s)
        scaled_s = pix_s.scaled(self.sample_img.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
        self.sample_img.setPixmap(scaled_s)

    def _build_weights_and_predict(self):
        pixels = self.sim.get_pixels()
        W = np.zeros((1024, self.n_classes), dtype=np.float32)
        for c in range(self.n_classes):
            W[:, c] = np.roll(pixels, shift=c * 64)
            
        x = self.X_test[self.current_sample_idx:self.current_sample_idx+1]
        logits = x @ W
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        pred = np.argmax(probs, axis=1)[0]
        conf = probs[0, pred]
        return W, pred, conf

    def update_visuals(self):
        W, pred, conf = self._build_weights_and_predict()
        
        w_min, w_max = np.min(W), np.max(W)
        if w_max - w_min < 1e-8:
            w_img = np.zeros_like(W)
        else:
            w_img = (W - w_min) / (w_max - w_min)
            
        idx_w = (w_img * 255).astype(np.uint8)
        rgb_w = np.ascontiguousarray(COLORMAPS["Jet"][idx_w])
        h_w, w_w = rgb_w.shape[:2]
        qimg_w = QImage(rgb_w.data, w_w, h_w, 3 * w_w, QImage.Format_RGB888).copy()
        pix_w = QPixmap.fromImage(qimg_w)
        scaled_w = pix_w.scaled(self.weight_img.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.weight_img.setPixmap(scaled_w)
        
        true_label = self.y_test[self.current_sample_idx]
        self.metrics_text.clear()
        self.metrics_text.append(f"<b>Sample:</b> {self.current_sample_idx}")
        self.metrics_text.append(f"<b>True Class:</b> {true_label}")
        self.metrics_text.append(f"<b>Predicted Class:</b> <span style='color:#2196F3; font-weight:bold;'>{pred}</span>")
        self.metrics_text.append(f"<b>Confidence:</b> {conf:.1%}")
        self.metrics_text.append(f"\n<i>Weights are live pixel values from Tab 1.</i>")
        self.metrics_text.append(f"<i>As the plasma animates, the weights shift!</i>")


# ========== Main Window ==========
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Animated Plasma -> Neural Network Weights")
        
        self.sim = PlasmaSim(width=32, height=32)
        
        self.tabs = QTabWidget()
        self.tab1 = PlasmaAnimationTab(self.sim)
        self.tab2 = NNWeightsTab(self.sim)
        
        self.tabs.addTab(self.tab1, "1. Animated Heatmap (Plasma)")
        self.tabs.addTab(self.tab2, "2. NN Weights from Pixels")
        
        self.setCentralWidget(self.tabs)
        
        self.tab1.pause_btn.clicked.connect(self._toggle_pause)
        self.tab1.reset_btn.clicked.connect(self._reset_sim)
        
        self.is_running = True
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(50)

    def _toggle_pause(self):
        self.is_running = not self.tab1.pause_btn.isChecked()
        self.tab1.pause_btn.setText("Resume" if not self.is_running else "Pause")

    def _reset_sim(self):
        self.sim.reset(new_seed=True)

    def _tick(self):
        if not self.is_running:
            return
            
        steps = self.tab1.speed_slider.value()
        for _ in range(steps):
            self.sim.step(dt=0.1)
            
        self.tab1.update_visuals()
        self.tab2.update_visuals()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.resize(950, 650)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()