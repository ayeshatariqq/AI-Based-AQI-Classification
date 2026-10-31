"""
Aether — Air Quality Classification (desktop application)
==========================================================
A native desktop app (PySide6 / Qt) around the trained AQI classifier:
XGBoost + SMOTE, hyper-parameter tuned, isotonic-calibrated.

Views
-----
  Single reading   Enter eight pollutant readings, get a category,
                   calibrated confidence, and plain-language health guidance.
  Batch analysis   Drop in a CSV; sortable / filterable results, category
                   breakdown, export. Rows that cannot be classified are
                   flagged with a reason instead of being silently guessed.
  History          Every single-reading analysis, kept between launches.
  About the model  Model card, test metrics, category legend, limitations.

Expects a `models/` folder next to this file containing the joblib artifacts
saved from the notebook:
    xgb_aqi_model.pkl, label_encoder.pkl, feature_columns.pkl,
    pollutant_columns.pkl, city_list.pkl, season_list.pkl, input_ranges.pkl (optional)
and a `fonts/` folder with the bundled Montserrat family. The desktop typography
uses a clean sans-serif treatment to stay visually aligned with the Aether web app.
"""

from __future__ import annotations

import calendar
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

APP_NAME = "Aether"
APP_VERSION = "1.0"


def resource_path(rel: str) -> Path:
    """Works both from source and from a PyInstaller bundle."""
    base = getattr(sys, "_MEIPASS", None)
    return (Path(base) if base else Path(__file__).resolve().parent) / rel


MODEL_DIR = resource_path("models")
FONT_DIR = resource_path("fonts")

# ---------------------------------------------------------------------------
# Model card — edit these if the model is retrained
# ---------------------------------------------------------------------------
MODEL_CARD = {
    "name": "XGBoost + SMOTE (tuned, calibrated)",
    "test_accuracy": 0.827,
    "test_macro_f1": 0.791,
    "train_data": "India city-level daily air quality, 2015\u20132020",
}

# ---------------------------------------------------------------------------
# Domain constants (CPCB AQI categories)
# ---------------------------------------------------------------------------
ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
SKIPPED = "Skipped"
PRED_COL = "Predicted_AQI_Bucket"

AQI_RANGE = {
    "Good": "AQI 0\u201350",
    "Satisfactory": "AQI 51\u2013100",
    "Moderate": "AQI 101\u2013200",
    "Poor": "AQI 201\u2013300",
    "Very Poor": "AQI 301\u2013400",
    "Severe": "AQI 401\u2013500",
}
AQI_HEALTH = {
    "Good": "Minimal impact on health.",
    "Satisfactory": "May cause minor breathing discomfort to sensitive people.",
    "Moderate": "May cause breathing discomfort to people with lung disease such as asthma, "
                "and discomfort to people with heart disease, children and older adults.",
    "Poor": "May cause breathing discomfort to most people on prolonged exposure, "
            "and discomfort to people with heart disease.",
    "Very Poor": "May cause respiratory illness on prolonged exposure; effects are more "
                 "pronounced in people with lung and heart disease.",
    "Severe": "May cause respiratory effects even in healthy people, and serious health "
              "impacts for people with lung or heart disease, even during light activity.",
}
UNITS = {"CO": "mg/m\u00b3"}          # everything else is \u00b5g/m\u00b3
DEFAULT_UNIT = "\u00b5g/m\u00b3"

# Illustrative inputs for trying the app out — NOT real measurements.
PRESETS = [
    ("Clean day", {"City": "Bengaluru", "Month": 8,
                   "PM2.5": 18, "PM10": 42, "NO": 5, "NO2": 14, "NOx": 12, "CO": 0.4, "SO2": 7, "O3": 28}),
    ("Moderate winter day", {"City": "Kolkata", "Month": 1,
                             "PM2.5": 90, "PM10": 160, "NO": 15, "NO2": 35, "NOx": 30, "CO": 1.1, "SO2": 12, "O3": 30}),
    ("Heavy smog episode", {"City": "Delhi", "Month": 11,
                            "PM2.5": 380, "PM10": 520, "NO": 45, "NO2": 85, "NOx": 95, "CO": 3.8, "SO2": 22, "O3": 55}),
]

# ---------------------------------------------------------------------------
# Visual identity
# ---------------------------------------------------------------------------
PAPER = "#f7f6f2"
RAISED = "#ffffff"
INK = "#222b27"
INK_SOFT = "#737b75"
HAIRLINE = "#dedfd8"
ACCENT = "#315c49"
ACCENT_DARK = "#244534"
ACCENT_SOFT = "#e9eee8"
WARN_BG = "#f8f1e2"
WARN_INK = "#765a22"
ERROR = "#a8432f"

SEVERITY = {          # muted, earth-toned — calm rather than traffic-light
    "Good": "#4f8762",
    "Satisfactory": "#82945b",
    "Moderate": "#b08a36",
    "Poor": "#bb713b",
    "Very Poor": "#a84f40",
    "Severe": "#7e3535",
    SKIPPED: "#9a948a",
}

FONT_DISPLAY = "Montserrat"
FONT_UI = "Montserrat"


def load_fonts() -> None:
    """Register bundled fonts; fall back gracefully if they are not present."""
    global FONT_DISPLAY, FONT_UI
    if FONT_DIR.is_dir():
        for f in sorted(FONT_DIR.glob("*.ttf")):
            QtGui.QFontDatabase.addApplicationFont(str(f))
    families = set(QtGui.QFontDatabase.families())
    if "Montserrat" in families:
        FONT_DISPLAY = "Montserrat"
        FONT_UI = "Montserrat"


def mkfont(family: str, px: int, weight=QtGui.QFont.Weight.Normal,
           italic: bool = False, spacing: float = 0.0) -> QtGui.QFont:
    f = QtGui.QFont(family)
    f.setPixelSize(px)
    f.setWeight(weight)
    f.setItalic(italic)
    if spacing:
        f.setLetterSpacing(QtGui.QFont.SpacingType.AbsoluteSpacing, spacing)
    return f


W = QtGui.QFont.Weight


def label(text: str = "", *, family=None, px=13, weight=W.Normal, italic=False,
          color=INK, spacing=0.0, wrap=False, upper=False) -> QtWidgets.QLabel:
    lb = QtWidgets.QLabel(text.upper() if upper else text)
    lb.setFont(mkfont(family or FONT_UI, px, weight, italic, spacing))
    lb.setStyleSheet(f"color:{color}; background:transparent;")
    lb.setWordWrap(wrap)
    return lb


def make_chevron() -> str:
    """Paint a small chevron PNG for combo boxes; returns a QSS-safe path."""
    d = Path(QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.TempLocation)) / "aether_assets"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "chevron.png"
    pm = QtGui.QPixmap(24, 24)
    pm.fill(Qt.GlobalColor.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    pen = QtGui.QPen(QtGui.QColor(INK_SOFT), 2.2)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.drawPolyline([QtCore.QPointF(6, 9), QtCore.QPointF(12, 15), QtCore.QPointF(18, 9)])
    p.end()
    pm.save(str(f))
    return f.as_posix()


def build_qss() -> str:
    chev = make_chevron()
    return f"""
    QWidget {{ color:{INK}; }}
    QMainWindow, QWidget#page, QScrollArea, QScrollArea > QWidget > QWidget {{ background:{PAPER}; }}
    QFrame#sidebar {{ background:{PAPER}; border-right:1px solid {HAIRLINE}; }}
    QFrame#card {{ background:{RAISED}; border:1px solid {HAIRLINE}; border-radius:7px; }}
    QFrame#dropzone {{ background:#fcfcfa; border:1px dashed #d9ddd5; border-radius:5px; }}
    QFrame#hair {{ background:{HAIRLINE}; border:none; max-height:1px; min-height:1px; }}
    QLabel#banner {{ background:{WARN_BG}; color:{WARN_INK}; border-radius:3px; padding:10px 14px; }}
    QLabel#banner[kind="ok"] {{ background:#e8efe4; color:#41603c; }}

    QPushButton#nav {{
        background:transparent; color:{INK_SOFT}; text-align:left;
        border:none; border-left:3px solid transparent; padding:12px 23px;
    }}
    QPushButton#nav:hover {{ color:{INK}; background:#f0f1eb; }}
    QPushButton#nav:checked {{ color:{ACCENT}; border-left:3px solid {ACCENT}; background:{ACCENT_SOFT}; }}

    QPushButton#primary {{
        background:{ACCENT}; color:#ffffff; border:1px solid {ACCENT}; border-radius:5px; padding:11px 18px;
    }}
    QPushButton#primary:hover {{ background:{ACCENT_DARK}; border-color:{ACCENT_DARK}; }}
    QPushButton#primary:disabled {{ background:#b9c4bb; border-color:#b9c4bb; }}
    QPushButton#ghost {{
        background:#ffffff; color:{INK_SOFT}; border:1px solid {HAIRLINE}; border-radius:5px; padding:11px 16px;
    }}
    QPushButton#ghost:hover {{ color:{INK}; border-color:{INK_SOFT}; }}
    QPushButton#ghost:disabled {{ color:#bdb7ab; border-color:#ece6da; }}

    QLineEdit, QComboBox {{
        background:#fdfdfb; border:1px solid {HAIRLINE}; border-radius:5px; padding:10px 12px;
        selection-background-color:{ACCENT_SOFT}; selection-color:{INK};
    }}
    QLineEdit:focus, QComboBox:focus {{ border-color:{ACCENT}; }}
    QComboBox::drop-down {{ border:none; width:30px; }}
    QComboBox::down-arrow {{ image:url({chev}); width:12px; height:12px; }}
    QLineEdit[error="true"] {{ border-color:{ERROR}; background:#fbefeb; }}
    QComboBox QAbstractItemView {{
        background:{RAISED}; border:1px solid {HAIRLINE}; outline:0; padding:4px;
        selection-background-color:{ACCENT_SOFT}; selection-color:{INK};
    }}

    QTableView {{
        background:{RAISED}; alternate-background-color:#fcfaf6; border:1px solid {HAIRLINE};
        border-radius:3px; gridline-color:transparent; outline:0;
        selection-background-color:{ACCENT_SOFT}; selection-color:{INK};
    }}
    QTableView::item {{ padding:6px 8px; border:none; }}
    QHeaderView::section {{
        background:{ACCENT_SOFT}; color:{INK_SOFT}; border:none; border-bottom:1px solid {HAIRLINE};
        padding:10px 10px;
    }}
    QTableCornerButton::section {{ background:{ACCENT_SOFT}; border:none; }}

    QScrollBar:vertical {{ background:transparent; width:12px; margin:2px; }}
    QScrollBar::handle:vertical {{ background:#d6dcd3; border-radius:4px; min-height:36px; }}
    QScrollBar::handle:vertical:hover {{ background:#b5c0b5; }}
    QScrollBar:horizontal {{ background:transparent; height:12px; margin:2px; }}
    QScrollBar::handle:horizontal {{ background:#d6dcd3; border-radius:4px; min-width:36px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width:0; height:0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background:transparent; }}

    QToolTip {{ background:{INK}; color:{PAPER}; border:none; padding:6px 9px; }}
    QMessageBox {{ background:{PAPER}; }}
    """


# ---------------------------------------------------------------------------
# Prediction engine — ONE code path for single, batch and history
# ---------------------------------------------------------------------------
class Engine:
    """Loads the saved artifacts and mirrors the notebook's preprocessing."""

    def __init__(self, model_dir: Path):
        self.model = joblib.load(model_dir / "xgb_aqi_model.pkl")
        self.le = joblib.load(model_dir / "label_encoder.pkl")
        self.feature_columns = list(joblib.load(model_dir / "feature_columns.pkl"))
        self.pollutants = list(joblib.load(model_dir / "pollutant_columns.pkl"))
        self.cities = sorted(joblib.load(model_dir / "city_list.pkl"))
        self.seasons = list(joblib.load(model_dir / "season_list.pkl"))
        classes = set(self.le.classes_)
        self.order = [c for c in ORDER if c in classes]
        # predict_proba columns follow model.classes_ (encoded ints) -> labels
        enc = np.asarray(self.model.classes_, dtype=int)
        self.proba_labels = list(self.le.inverse_transform(enc))
        # Optional training-distribution metadata; absence never affects prediction.
        ranges_path = model_dir / "input_ranges.pkl"
        self.input_ranges = joblib.load(ranges_path) if ranges_path.is_file() else {}

    def warnings_for_reading(self, readings: dict) -> list[str]:
        """Return soft distribution warnings without changing classification."""
        warnings = []
        if not isinstance(self.input_ranges, dict):
            return warnings
        for pollutant, value in readings.items():
            limits = self.input_ranges.get(pollutant)
            if not isinstance(limits, dict):
                continue
            try:
                numeric = float(value)
                maximum = float(limits["max"])
            except (KeyError, TypeError, ValueError):
                continue
            if numeric > maximum:
                warnings.append(
                    f"{pollutant} = {numeric:g} is above the highest value seen in training "
                    f"({maximum:g}). Check the units; the prediction may be unreliable."
                )
        ozone = self.input_ranges.get("O3")
        if isinstance(ozone, dict):
            try:
                if float(readings.get("O3", float("-inf"))) > float(ozone["p99"]):
                    warnings.append(
                        "Ozone is unusually high for the training data. The model is least "
                        "reliable when ozone, rather than particulate matter, drives the AQI."
                    )
            except (KeyError, TypeError, ValueError):
                pass
        return warnings

    @staticmethod
    def season(month: int) -> str:
        if month in (12, 1, 2):
            return "Winter"
        if month in (3, 4, 5):
            return "Summer"
        if month in (6, 7, 8, 9):
            return "Monsoon"
        return "Post-Monsoon"

    def predict_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Classify every valid row; rows that can't be classified are kept,
        marked 'Skipped', with the reason in the Note column."""
        df = raw.copy()
        # CSVs often contain a UTF-8 BOM, padded headers, or different header casing.
        # Normalize these harmless variations before checking the schema.
        cleaned = {}
        expected = {str(c).strip().casefold(): str(c).strip() for c in ["City", "Date", "Month", *self.pollutants]}
        for col in df.columns:
            original = str(col).replace("\ufeff", "").strip()
            canonical = expected.get(original.casefold(), original)
            cleaned[col] = canonical
        df = df.rename(columns=cleaned)
        # Handle common spacing variants such as "PM 2.5" when the model expects "PM2.5".
        normalized_lookup = {"".join(ch for ch in str(c).casefold() if ch.isalnum()): c for c in df.columns}
        rename = {}
        for required in ["City", "Date", "Month", *self.pollutants]:
            key = "".join(ch for ch in required.casefold() if ch.isalnum())
            if required not in df.columns and key in normalized_lookup:
                rename[normalized_lookup[key]] = required
        if rename:
            df = df.rename(columns=rename)

        if "City" not in df.columns:
            raise ValueError("The CSV needs a 'City' column. Use the Template button to download an example.")
        missing = [c for c in self.pollutants if c not in df.columns]
        if missing:
            raise ValueError("Missing required pollutant column(s): " + ", ".join(missing) + ". Use the Template button for the exact format.")
        if "Date" in df.columns:
            month = pd.to_datetime(df["Date"], errors="coerce").dt.month
        elif "Month" in df.columns:
            month = pd.to_numeric(df["Month"], errors="coerce")
        else:
            raise ValueError("The file needs a 'Date' or a 'Month' column.")

        nums = df[self.pollutants].apply(pd.to_numeric, errors="coerce")
        city_input = df["City"].astype(str).str.strip()
        city_lookup = {str(c).casefold(): str(c) for c in self.cities}
        city = city_input.map(lambda value: city_lookup.get(value.casefold(), value))

        note = pd.Series("", index=df.index, dtype=object)
        note[~city.isin(self.cities)] = "City not supported by the model"
        note[(note == "") & ~month.between(1, 12)] = "Invalid or missing date / month"
        note[(note == "") & nums.isna().any(axis=1)] = "Missing or non-numeric reading"
        note[(note == "") & (nums < 0).any(axis=1)] = "Negative reading"
        ok = note == ""

        out = df.copy()
        out[PRED_COL] = SKIPPED
        out["Confidence"] = np.nan
        for c in self.order:
            out["P_" + c] = np.nan
        out["Note"] = note

        if ok.any():
            X = nums[ok].copy()
            m_ok = month[ok].astype(int)
            X["Month"] = m_ok
            dummies = pd.get_dummies(
                pd.DataFrame({"City": city[ok], "Season": m_ok.map(self.season)}), dtype=float)
            X = pd.concat([X, dummies], axis=1).reindex(columns=self.feature_columns, fill_value=0).astype(float)

            proba = np.asarray(self.model.predict_proba(X))
            labels = np.array(self.proba_labels)[proba.argmax(axis=1)]
            out.loc[ok, PRED_COL] = labels
            out.loc[ok, "Confidence"] = proba.max(axis=1)
            for j, lab in enumerate(self.proba_labels):
                if "P_" + lab in out.columns:
                    out.loc[ok, "P_" + lab] = proba[:, j]
        return out


# ---------------------------------------------------------------------------
# Reusable widgets
# ---------------------------------------------------------------------------
class Card(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")


def hline() -> QtWidgets.QFrame:
    f = QtWidgets.QFrame()
    f.setObjectName("hair")
    f.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    return f


def page_header(eyebrow: str, title: str, subtitle: str = "") -> QtWidgets.QWidget:
    w = QtWidgets.QWidget()
    v = QtWidgets.QVBoxLayout(w)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(2)
    v.addWidget(label(eyebrow, px=11, weight=W.Medium, color=ACCENT, spacing=2.2, upper=True))
    v.addWidget(label(title, family=FONT_DISPLAY, px=36, weight=W.Bold))
    if subtitle:
        s = label(subtitle, px=13, weight=W.Light, color=INK_SOFT, wrap=True)
        v.addSpacing(4)
        v.addWidget(s)
    return w


def primary_button(text: str, ghost=False) -> QtWidgets.QPushButton:
    b = QtWidgets.QPushButton(text)
    b.setObjectName("ghost" if ghost else "primary")
    b.setFont(mkfont(FONT_UI, 12, W.DemiBold))
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


class ProbabilityBars(QtWidgets.QWidget):
    """Horizontal bars drawn directly with QPainter — crisp, animated, light."""
    ROW_H = 30

    def __init__(self, order, parent=None):
        super().__init__(parent)
        self.order = list(order)
        self.values = {c: 0.0 for c in self.order}
        self.scale = 1.0
        self.highlight = None
        self.formatter = lambda v: f"{v * 100:.0f}%"
        self._t = 1.0
        self._anim = QtCore.QVariantAnimation(self)
        self._anim.setDuration(520)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._tick)
        self.setMinimumHeight(self.ROW_H * len(self.order) + 4)

    def sizeHint(self):
        return QtCore.QSize(320, self.ROW_H * len(self.order) + 4)

    def _tick(self, v):
        self._t = float(v)
        self.update()

    def set_values(self, values: dict, highlight=None, scale=None, formatter=None, animate=True):
        self.values = {c: float(values.get(c, 0.0) or 0.0) for c in self.order}
        self.highlight = highlight
        self.scale = scale if scale else 1.0
        if formatter:
            self.formatter = formatter
        if animate:
            self._anim.stop()
            self._anim.start()
        else:
            self._t = 1.0
            self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        label_w, val_w, gap = 104, 54, 12
        track_x = label_w + gap
        track_w = max(20, self.width() - track_x - val_w - gap)
        al_l = Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        al_r = Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight
        for i, name in enumerate(self.order):
            y = i * self.ROW_H
            cy = y + self.ROW_H / 2
            v = self.values.get(name, 0.0)
            frac = min(max(v / self.scale, 0.0), 1.0) * self._t if self.scale else 0.0
            hl = name == self.highlight
            p.setFont(mkfont(FONT_UI, 12, W.Medium if hl else W.Normal))
            p.setPen(QtGui.QColor(INK if hl else INK_SOFT))
            p.drawText(QtCore.QRectF(0, y, label_w, self.ROW_H), al_l, name)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QtGui.QColor(ACCENT_SOFT))
            p.drawRoundedRect(QtCore.QRectF(track_x, cy - 3, track_w, 6), 3, 3)
            if self.scale and v / self.scale >= 0.005:
                p.setBrush(QtGui.QColor(SEVERITY.get(name, INK_SOFT)))
                p.drawRoundedRect(QtCore.QRectF(track_x, cy - 3, max(6.0, track_w * frac), 6), 3, 3)
            p.setPen(QtGui.QColor(INK if hl else INK_SOFT))
            p.setFont(mkfont(FONT_UI, 11, W.Medium if hl else W.Normal))
            p.drawText(QtCore.QRectF(self.width() - val_w, y, val_w, self.ROW_H), al_r, self.formatter(v))


class Dot(QtWidgets.QLabel):
    def __init__(self, color=HAIRLINE, size=14, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.set_color(color, size)

    def set_color(self, color, size=None):
        s = size or self.width()
        self.setStyleSheet(f"background:{color}; border-radius:{s // 2}px;")


# ---- table model / delegate -------------------------------------------------
class FrameModel(QtCore.QAbstractTableModel):
    """Read-only model over a DataFrame with sensible formatting and sorting."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.df = pd.DataFrame()
        self.cols: list[str] = []
        self.note = None
        self.cities: list[str] = []
        self.preds: list[str] = []

    def set_frame(self, df: pd.DataFrame, cols: list[str]):
        self.beginResetModel()
        self.df = df.reset_index(drop=True)
        self.cols = [c for c in cols if c in self.df.columns]
        self.note = self.df["Note"].tolist() if "Note" in self.df.columns else None
        self.cities = self.df["City"].astype(str).str.lower().tolist() if "City" in self.df.columns else []
        self.preds = self.df[PRED_COL].astype(str).tolist() if PRED_COL in self.df.columns else []
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.df)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.cols)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and section < len(self.cols):
            name = self.cols[section]
            if role == Qt.ItemDataRole.DisplayRole:
                return ("Predicted" if name == PRED_COL else name).upper()
            if role == Qt.ItemDataRole.FontRole:
                return mkfont(FONT_UI, 10, W.Medium, spacing=0.8)
        return None

    @staticmethod
    def _fmt(v):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return "\u2014"
            return ("%.2f" % v).rstrip("0").rstrip(".")
        return str(v)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        name = self.cols[index.column()]
        v = self.df.iat[index.row(), self.df.columns.get_loc(name)]
        if role == Qt.ItemDataRole.DisplayRole:
            if name == "Confidence":
                return "\u2014" if pd.isna(v) else f"{v * 100:.1f}%"
            if name == "Date" and not isinstance(v, str) and not pd.isna(v):
                return str(v)[:10]
            return self._fmt(v)
        if role == Qt.ItemDataRole.UserRole:     # sort key
            if name == PRED_COL:
                return ORDER.index(v) if v in ORDER else 99
            if isinstance(v, (int, float, np.integer, np.floating)):
                return float(v) if not pd.isna(v) else -1.0
            return str(v).lower()
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if name == PRED_COL:
                return int(Qt.AlignmentFlag.AlignCenter)
            if isinstance(v, (int, float, np.integer, np.floating)):
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.FontRole:
            return mkfont(FONT_UI, 12)
        if role == Qt.ItemDataRole.ToolTipRole and name == PRED_COL and self.note:
            n = self.note[index.row()]
            return n or None
        return None


class PillDelegate(QtWidgets.QStyledItemDelegate):
    def paint(self, painter, option, index):
        text = index.data(Qt.ItemDataRole.DisplayRole)
        if text not in SEVERITY:                      # ordinary cell -> default painting
            super().paint(painter, option, index)
            return
        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        style = opt.widget.style() if opt.widget else QtWidgets.QApplication.style()
        style.drawControl(QtWidgets.QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget)
        color = SEVERITY.get(text)
        if not color:
            return
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        f = mkfont(FONT_UI, 11, W.Medium)
        painter.setFont(f)
        tw = QtGui.QFontMetrics(f).horizontalAdvance(text)
        w, h = tw + 22, 22
        r = QtCore.QRectF(0, 0, w, h)
        r.moveCenter(QtCore.QRectF(option.rect).center())
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(color))
        painter.drawRoundedRect(r, 11, 11)
        painter.setPen(QtGui.QColor("#ffffff"))
        painter.drawText(r, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()

    def sizeHint(self, option, index):
        s = super().sizeHint(option, index)
        return QtCore.QSize(s.width(), max(s.height(), 34))


class FilterProxy(QtCore.QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cat = "All categories"
        self.text = ""
        self.setSortRole(Qt.ItemDataRole.UserRole)

    def set_filters(self, cat: str, text: str):
        self.cat, self.text = cat, text.strip().lower()
        self.invalidateFilter()

    def filterAcceptsRow(self, row, parent):
        m = self.sourceModel()
        if m is None:
            return True
        if self.cat != "All categories" and row < len(m.preds) and m.preds[row] != self.cat:
            return False
        if self.text and row < len(m.cities) and self.text not in m.cities[row]:
            return False
        return True


def make_table(model: QtCore.QAbstractItemModel, pill_col: int | None) -> tuple[QtWidgets.QTableView, FilterProxy]:
    proxy = FilterProxy()
    proxy.setSourceModel(model)
    view = QtWidgets.QTableView()
    view.setModel(proxy)
    view.setAlternatingRowColors(True)
    view.setShowGrid(False)
    view.setSortingEnabled(True)
    view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
    view.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    view.verticalHeader().setVisible(False)
    view.verticalHeader().setDefaultSectionSize(34)
    view.horizontalHeader().setHighlightSections(False)
    view.horizontalHeader().setStretchLastSection(True)
    view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    view.setItemDelegate(PillDelegate(view))
    return view, proxy


def fit_columns(view: QtWidgets.QTableView):
    hh = view.horizontalHeader()
    hh.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
    hh.setStretchLastSection(True)


# ---------------------------------------------------------------------------
# Page: single reading
# ---------------------------------------------------------------------------
class SinglePage(QtWidgets.QWidget):
    analyzed = QtCore.Signal(dict)

    def __init__(self, engine: Engine):
        super().__init__()
        self.setObjectName("page")
        self.engine = engine
        self.edits: dict[str, QtWidgets.QLineEdit] = {}

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(44, 36, 44, 30)
        root.setSpacing(24)
        root.addWidget(page_header(
            "Single reading", "Read the air.",
            "Enter one day\u2019s average pollutant readings for a city. The model returns an AQI "
            "category with calibrated confidence."))

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(24)
        root.addLayout(body, 1)

        # ---- form card
        form = Card()
        form.setFixedWidth(470)
        fv = QtWidgets.QVBoxLayout(form)
        fv.setContentsMargins(28, 24, 28, 24)
        fv.setSpacing(0)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(label("Input measurements", family=FONT_DISPLAY, px=19, weight=W.DemiBold))
        top.addStretch()
        self.preset = QtWidgets.QComboBox()
        self.preset.setFont(mkfont(FONT_UI, 12))
        self.preset.addItem("Try an example\u2026")
        for name, _ in PRESETS:
            self.preset.addItem(name)
        self.preset.setCursor(Qt.CursorShape.PointingHandCursor)
        self.preset.setToolTip("Illustrative values for trying the app \u2014 not real measurements")
        self.preset.activated.connect(self.load_preset)
        top.addWidget(self.preset)
        fv.addLayout(top)
        fv.addSpacing(2)
        fv.addWidget(label("Daily averages. CO in mg/m\u00b3, all others in \u00b5g/m\u00b3.",
                           px=11, weight=W.Light, color=INK_SOFT))
        fv.addWidget(label(
            "Enter CO and O3 as daily averages. Official AQI uses 8-hour maxima for these two, "
            "which run higher; entering 8-hour maxima will push the prediction up.",
            px=11, weight=W.Light, color=INK_SOFT, wrap=True))
        fv.addSpacing(18)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(14)
        for i, p in enumerate(engine.pollutants):
            e = QtWidgets.QLineEdit()
            e.setFont(mkfont(FONT_UI, 14))
            e.setPlaceholderText("0.0")
            val = QtGui.QDoubleValidator(0.0, 100000.0, 3, e)
            val.setNotation(QtGui.QDoubleValidator.Notation.StandardNotation)
            val.setLocale(QtCore.QLocale.c())
            e.setValidator(val)
            e.returnPressed.connect(self.analyze)
            e.textChanged.connect(lambda _=None, ed=e: self._clear_error(ed))
            self.edits[p] = e
            grid.addWidget(self._field(p.upper(), e, unit=UNITS.get(p, DEFAULT_UNIT)), i // 2, i % 2)
        fv.addLayout(grid)
        fv.addSpacing(16)
        fv.addWidget(hline())
        fv.addSpacing(16)

        g2 = QtWidgets.QGridLayout()
        g2.setHorizontalSpacing(18)
        self.city = QtWidgets.QComboBox()
        self.city.setFont(mkfont(FONT_UI, 14))
        self.city.addItems(engine.cities)
        self.month = QtWidgets.QComboBox()
        self.month.setFont(mkfont(FONT_UI, 14))
        for m in range(1, 13):
            self.month.addItem(calendar.month_name[m], m)
        self.month.setCurrentIndex(datetime.now().month - 1)
        g2.addWidget(self._field("City", self.city), 0, 0)
        g2.addWidget(self._field("Month", self.month), 0, 1)
        fv.addLayout(g2)
        fv.addSpacing(6)

        self.error = label("", px=12, weight=W.Medium, color=ERROR, wrap=True)
        self.error.hide()
        fv.addSpacing(8)
        fv.addWidget(self.error)
        fv.addStretch()
        fv.addWidget(label("Press Ctrl + Enter to analyze.", px=11, weight=W.Light, color=INK_SOFT))
        fv.addSpacing(12)

        btns = QtWidgets.QHBoxLayout()
        self.go = primary_button("Analyze")
        self.go.clicked.connect(self.analyze)
        clear = primary_button("Clear", ghost=True)
        clear.clicked.connect(self.clear)
        btns.addWidget(self.go, 1)
        btns.addWidget(clear)
        fv.addLayout(btns)
        body.addWidget(form)

        # ---- result card
        res = Card()
        rv = QtWidgets.QVBoxLayout(res)
        rv.setContentsMargins(34, 28, 34, 28)
        rv.setSpacing(0)
        rv.addWidget(label("Result", px=11, weight=W.Medium, color=INK_SOFT, spacing=2.0, upper=True))
        rv.addSpacing(10)

        self.empty = QtWidgets.QFrame()
        self.empty.setObjectName("dropzone")
        ev = QtWidgets.QVBoxLayout(self.empty)
        ev.setContentsMargins(24, 60, 24, 60)
        ev.addWidget(label("Enter readings and press Analyze\nto see a classification.",
                           px=13, weight=W.Light, color=INK_SOFT), 0, Qt.AlignmentFlag.AlignCenter)
        rv.addWidget(self.empty)

        self.filled = QtWidgets.QWidget()
        fl = QtWidgets.QVBoxLayout(self.filled)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(0)
        head = QtWidgets.QHBoxLayout()
        head.setSpacing(16)
        self.dot = Dot(HAIRLINE, 16)
        head.addWidget(self.dot, 0, Qt.AlignmentFlag.AlignVCenter)
        self.cat_label = label("\u2014", family=FONT_DISPLAY, px=48, weight=W.Bold)
        head.addWidget(self.cat_label)
        head.addStretch()
        fl.addLayout(head)
        self.meta = label("", px=11, weight=W.Medium, color=INK_SOFT, spacing=1.4, upper=True)
        fl.addWidget(self.meta)
        fl.addSpacing(14)
        self.health = label("", px=14, weight=W.Light, color=INK, wrap=True)
        fl.addWidget(self.health)
        self.range_warning = QtWidgets.QLabel()
        self.range_warning.setObjectName("banner")
        self.range_warning.setFont(mkfont(FONT_UI, 11, W.Medium))
        self.range_warning.setWordWrap(True)
        self.range_warning.hide()
        fl.addWidget(self.range_warning)
        fl.addSpacing(10)
        self.borderline = label("", px=12, weight=W.Medium, color=WARN_INK, wrap=True)
        fl.addWidget(self.borderline)
        fl.addSpacing(18)
        fl.addWidget(hline())
        fl.addSpacing(18)
        fl.addWidget(label("Probability by category", px=11, weight=W.Medium, color=INK_SOFT,
                           spacing=2.0, upper=True))
        fl.addSpacing(8)
        self.bars = ProbabilityBars(engine.order)
        fl.addWidget(self.bars)
        fl.addStretch()
        self.context = label("", px=11, weight=W.Light, color=INK_SOFT, wrap=True)
        fl.addWidget(self.context)
        self.filled.hide()
        rv.addWidget(self.filled, 1)
        rv.addStretch()
        body.addWidget(res, 1)

        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Return"), self, activated=self.analyze)

    # -- helpers
    @staticmethod
    def _field(caption: str, widget: QtWidgets.QWidget, unit: str = "") -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        text = caption.upper() + (f"   \u00b7   {unit}" if unit else "")   # unit keeps its own case
        v.addWidget(label(text, px=10, weight=W.Medium, color=INK_SOFT, spacing=1.2))
        v.addWidget(widget)
        return w

    def _clear_error(self, ed):
        if ed.property("error"):
            ed.setProperty("error", False)
            ed.style().unpolish(ed)
            ed.style().polish(ed)

    def _mark_error(self, ed):
        ed.setProperty("error", True)
        ed.style().unpolish(ed)
        ed.style().polish(ed)

    def load_preset(self, idx: int):
        if idx <= 0:
            return
        _, vals = PRESETS[idx - 1]
        for p, e in self.edits.items():
            if p in vals:
                e.setText(str(vals[p]))
        if vals["City"] in self.engine.cities:
            self.city.setCurrentText(vals["City"])
        self.month.setCurrentIndex(vals["Month"] - 1)
        self.preset.setCurrentIndex(0)
        self.analyze()

    def clear(self):
        for e in self.edits.values():
            e.clear()
            self._clear_error(e)
        self.error.hide()
        self.range_warning.hide()
        self.filled.hide()
        self.empty.show()

    def analyze(self):
        vals, missing = {}, []
        for p, e in self.edits.items():
            t = e.text().strip()
            if not t:
                missing.append(p)
                self._mark_error(e)
                continue
            try:
                vals[p] = float(t)
            except ValueError:
                missing.append(p)
                self._mark_error(e)
        if missing:
            self.error.setText("Please enter a value for: " + ", ".join(missing) + ".")
            self.error.show()
            return
        self.error.hide()

        month = int(self.month.currentData())
        city = self.city.currentText()
        res = self.engine.predict_frame(pd.DataFrame([{**vals, "City": city, "Month": month}])).iloc[0]
        cat = res[PRED_COL]
        probs = {c: float(res.get("P_" + c, 0.0)) for c in self.engine.order}
        conf = float(res["Confidence"])
        color = SEVERITY[cat]

        self.dot.set_color(color, 16)
        self.cat_label.setText(cat)
        self.cat_label.setStyleSheet(f"color:{color}; background:transparent;")
        self.meta.setText(f"{AQI_RANGE[cat]}   \u00b7   {conf * 100:.1f}% confidence")
        self.health.setText(AQI_HEALTH[cat])
        range_warnings = self.engine.warnings_for_reading(vals)
        self.range_warning.setText("  ".join(range_warnings))
        self.range_warning.setVisible(bool(range_warnings))

        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        runner, rp = ranked[1]
        if conf < 0.60 or rp >= 0.25:
            self.borderline.setText(
                f"Borderline reading \u2014 {runner} is also plausible ({rp * 100:.0f}%). "
                f"Treat the category as approximate.")
            self.borderline.show()
        else:
            self.borderline.hide()

        self.bars.set_values(probs, highlight=cat, scale=1.0)
        self.context.setText(f"Based on daily-average readings for {city} in {calendar.month_name[month]}.")
        self.empty.hide()
        self.filled.show()

        rec = {"Time": datetime.now().strftime("%Y-%m-%d %H:%M"), "City": city, "Month": month, **vals,
               PRED_COL: cat, "Confidence": conf}
        self.analyzed.emit(rec)


# ---------------------------------------------------------------------------
# Page: batch analysis
# ---------------------------------------------------------------------------
class BatchPage(QtWidgets.QWidget):
    def __init__(self, engine: Engine):
        super().__init__()
        self.setObjectName("page")
        self.engine = engine
        self.result: pd.DataFrame | None = None
        self.setAcceptDrops(True)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(44, 36, 44, 30)
        root.setSpacing(18)

        head = QtWidgets.QHBoxLayout()
        head.addWidget(page_header("Batch analysis", "Many days at once.",
                                   "Open or drop a CSV of daily readings. Every row is classified; "
                                   "rows the model can\u2019t handle are flagged, not guessed."), 1)
        btns = QtWidgets.QVBoxLayout()
        btns.addStretch()
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(10)
        self.b_template = primary_button("Template", ghost=True)
        self.b_template.setToolTip("Save an example CSV with the required columns")
        self.b_template.clicked.connect(self.save_template)
        self.b_open = primary_button("Open CSV")
        self.b_open.clicked.connect(self.open_dialog)
        self.b_export = primary_button("Export", ghost=True)
        self.b_export.setEnabled(False)
        self.b_export.clicked.connect(self.export)
        for b in (self.b_template, self.b_export, self.b_open):
            row.addWidget(b)
        btns.addLayout(row)
        head.addLayout(btns)
        root.addLayout(head)
        root.addWidget(label(
            "Enter CO and O3 as daily averages. Official AQI uses 8-hour maxima for these two, "
            "which run higher; entering 8-hour maxima will push the prediction up.",
            px=11, weight=W.Light, color=INK_SOFT, wrap=True))

        self.banner = QtWidgets.QLabel()
        self.banner.setObjectName("banner")
        self.banner.setFont(mkfont(FONT_UI, 12, W.Medium))
        self.banner.setWordWrap(True)
        self.banner.hide()
        root.addWidget(self.banner)

        # empty state / drop zone
        self.drop = QtWidgets.QFrame()
        self.drop.setObjectName("dropzone")
        dv = QtWidgets.QVBoxLayout(self.drop)
        dv.setContentsMargins(40, 70, 40, 70)
        dv.setSpacing(10)
        dv.addWidget(label("Drop a CSV file here", family=FONT_DISPLAY, px=24, weight=W.DemiBold),
                     0, Qt.AlignmentFlag.AlignCenter)
        dv.addWidget(label("or use Open CSV.  Required columns:", px=13, weight=W.Light, color=INK_SOFT),
                     0, Qt.AlignmentFlag.AlignCenter)
        cols = ["City", "Date (or Month)"] + engine.pollutants
        dv.addWidget(label("  \u00b7  ".join(cols), px=12, weight=W.Medium, color=ACCENT, wrap=True),
                     0, Qt.AlignmentFlag.AlignCenter)
        dv.addStretch()
        root.addWidget(self.drop, 1)

        # results
        self.results = QtWidgets.QWidget()
        rl = QtWidgets.QHBoxLayout(self.results)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(20)

        left = QtWidgets.QVBoxLayout()
        left.setSpacing(10)
        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(10)
        self.file_label = label("", px=12, weight=W.Medium, color=INK_SOFT)
        bar.addWidget(self.file_label, 1)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search city\u2026")
        self.search.setFont(mkfont(FONT_UI, 12))
        self.search.setFixedWidth(170)
        self.search.setClearButtonEnabled(True)
        self.catbox = QtWidgets.QComboBox()
        self.catbox.setFont(mkfont(FONT_UI, 12))
        self.catbox.addItems(["All categories"] + engine.order + [SKIPPED])
        self.catbox.setFixedWidth(170)
        bar.addWidget(self.search)
        bar.addWidget(self.catbox)
        left.addLayout(bar)

        self.model = FrameModel(self)
        self.table, self.proxy = make_table(self.model, None)
        left.addWidget(self.table, 1)
        rl.addLayout(left, 1)

        side = Card()
        side.setFixedWidth(310)
        sv = QtWidgets.QVBoxLayout(side)
        sv.setContentsMargins(24, 22, 24, 22)
        sv.setSpacing(0)
        sv.addWidget(label("Breakdown", family=FONT_DISPLAY, px=19, weight=W.DemiBold))
        sv.addSpacing(10)
        self.bars = ProbabilityBars(engine.order)
        sv.addWidget(self.bars)
        sv.addSpacing(16)
        sv.addWidget(hline())
        sv.addSpacing(14)
        self.stats = {}
        for key, cap in (("total", "Rows"), ("done", "Classified"), ("skipped", "Skipped"),
                         ("poor", "Poor or worse"), ("top", "Most common")):
            r = QtWidgets.QHBoxLayout()
            r.addWidget(label(cap, px=10, weight=W.Medium, color=INK_SOFT, spacing=1.2, upper=True))
            r.addStretch()
            v = label("\u2014", px=13, weight=W.Medium)
            self.stats[key] = v
            r.addWidget(v)
            sv.addLayout(r)
            sv.addSpacing(10)
        sv.addStretch()
        rl.addWidget(side)
        self.results.hide()
        root.addWidget(self.results, 1)

        self.search.textChanged.connect(self._filter)
        self.catbox.currentTextChanged.connect(self._filter)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+O"), self, activated=self.open_dialog)

    # -- drag & drop
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and any(u.toLocalFile().lower().endswith(".csv") for u in e.mimeData().urls()):
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            if u.toLocalFile().lower().endswith(".csv"):
                self.load(u.toLocalFile())
                break

    # -- actions
    def open_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV files (*.csv)")
        if path:
            self.load(path)

    def load(self, path: str):
        try:
            QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                raw = pd.read_csv(path, encoding="utf-8-sig", sep=None, engine="python")
            except UnicodeDecodeError:
                raw = pd.read_csv(path, encoding="latin-1", sep=None, engine="python")
            if raw.empty or len(raw.columns) == 0:
                raise ValueError("The file contains no data rows. Check that this is a CSV with a header row.")
            res = self.engine.predict_frame(raw)
            for row_idx in res.index[res[PRED_COL] != SKIPPED]:
                readings = {p: res.at[row_idx, p] for p in self.engine.pollutants}
                advisory = self.engine.warnings_for_reading(readings)
                if advisory:
                    existing = str(res.at[row_idx, "Note"]) if "Note" in res.columns else ""
                    res.at[row_idx, "Note"] = " ".join(x for x in (existing, " ".join(advisory)) if x)
        except Exception as ex:
            QtWidgets.QApplication.restoreOverrideCursor()
            self._banner(f"Couldn\u2019t process that file: {ex}", ok=False)
            return
        QtWidgets.QApplication.restoreOverrideCursor()

        self.result = res
        shown = [c for c in ("City", "Date", "Month") if c in res.columns] + [PRED_COL, "Confidence"] \
                + self.engine.pollutants + (["Note"] if "Note" in res.columns else [])
        self.model.set_frame(res, shown)
        pill = self.model.cols.index(PRED_COL)
        self.table.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
        fit_columns(self.table)
        self.table.setColumnWidth(pill, 130)
        self.table.horizontalHeader().setSectionResizeMode(pill, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.search.clear()
        self.catbox.setCurrentIndex(0)
        self._filter()

        done = res[res[PRED_COL] != SKIPPED]
        n, k = len(res), len(done)
        counts = done[PRED_COL].value_counts().to_dict()
        mx = max(counts.values()) if counts else 1
        self.bars.set_values(counts, scale=mx, formatter=lambda v: f"{int(v)}")
        worse = sum(counts.get(c, 0) for c in ("Poor", "Very Poor", "Severe"))
        self.stats["total"].setText(f"{n:,}")
        self.stats["done"].setText(f"{k:,}")
        self.stats["skipped"].setText(f"{n - k:,}")
        self.stats["poor"].setText(f"{worse / k * 100:.0f}%" if k else "\u2014")
        self.stats["top"].setText(max(counts, key=counts.get) if counts else "\u2014")
        self.file_label.setText(Path(path).name)

        if n - k:
            reasons = res.loc[res[PRED_COL] == SKIPPED, "Note"].value_counts()
            detail = "; ".join(f"{v} \u00d7 {r.lower()}" for r, v in reasons.items())
            self._banner(f"{n - k} of {n} rows were skipped \u2014 {detail}. Hover a \u201cSkipped\u201d "
                         f"tag for details.", ok=False)
        else:
            self._banner(f"All {n:,} rows classified.", ok=True)
        self.drop.hide()
        self.results.show()
        self.b_export.setEnabled(True)

    def _banner(self, text: str, ok: bool):
        self.banner.setText(text)
        self.banner.setProperty("kind", "ok" if ok else "warn")
        self.banner.style().unpolish(self.banner)
        self.banner.style().polish(self.banner)
        self.banner.show()

    def _filter(self, *_):
        self.proxy.set_filters(self.catbox.currentText(), self.search.text())

    def export(self):
        if self.result is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export results", "aether_predictions.csv",
                                                        "CSV files (*.csv)")
        if path:
            out = self.result.copy()
            out.to_csv(path, index=False)
            self._banner(f"Exported {len(out):,} rows to {Path(path).name}.", ok=True)

    def save_template(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save template", "aether_template.csv",
                                                        "CSV files (*.csv)")
        if not path:
            return
        rows = []
        for i, (_, v) in enumerate(PRESETS[:2]):
            c = v["City"] if v["City"] in self.engine.cities else self.engine.cities[0]
            rows.append({"City": c, "Date": f"2020-0{i + 1}-15", **{p: v.get(p, 0) for p in self.engine.pollutants}})
        pd.DataFrame(rows).to_csv(path, index=False)
        self._banner(f"Template saved to {Path(path).name} (illustrative values \u2014 replace with your data).",
                     ok=True)


# ---------------------------------------------------------------------------
# Page: history
# ---------------------------------------------------------------------------
class HistoryPage(QtWidgets.QWidget):
    def __init__(self, engine: Engine):
        super().__init__()
        self.setObjectName("page")
        self.engine = engine
        base = Path(QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.AppDataLocation))
        self.path = base / "history.csv"
        self.cols = ["Time", "City", "Month", PRED_COL, "Confidence"] + engine.pollutants
        self.df = self._read()

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(44, 36, 44, 30)
        root.setSpacing(18)
        head = QtWidgets.QHBoxLayout()
        head.addWidget(page_header("History", "What you\u2019ve checked.",
                                   "Single-reading analyses are kept between sessions on this computer."), 1)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(10)
        self.b_export = primary_button("Export", ghost=True)
        self.b_export.clicked.connect(self.export)
        self.b_clear = primary_button("Clear history", ghost=True)
        self.b_clear.clicked.connect(self.clear)
        row.addWidget(self.b_export)
        row.addWidget(self.b_clear)
        col = QtWidgets.QVBoxLayout()
        col.addStretch()
        col.addLayout(row)
        head.addLayout(col)
        root.addLayout(head)

        self.empty = QtWidgets.QFrame()
        self.empty.setObjectName("dropzone")
        ev = QtWidgets.QVBoxLayout(self.empty)
        ev.setContentsMargins(40, 70, 40, 70)
        ev.addWidget(label("Nothing here yet.", family=FONT_DISPLAY, px=22, weight=W.DemiBold),
                     0, Qt.AlignmentFlag.AlignCenter)
        ev.addWidget(label("Analyses from the Single reading view will appear here.", px=13, weight=W.Light,
                           color=INK_SOFT), 0, Qt.AlignmentFlag.AlignCenter)
        ev.addStretch()
        root.addWidget(self.empty, 1)

        self.model = FrameModel(self)
        self.table, self.proxy = make_table(self.model, None)
        root.addWidget(self.table, 1)
        self._refresh()

    def _read(self) -> pd.DataFrame:
        try:
            if self.path.exists():
                d = pd.read_csv(self.path)
                if all(c in d.columns for c in self.cols):
                    return d[self.cols]
        except Exception:
            pass
        return pd.DataFrame(columns=self.cols)

    def _save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.df.to_csv(self.path, index=False)
        except OSError:
            pass

    def _refresh(self):
        has = len(self.df) > 0
        self.empty.setVisible(not has)
        self.table.setVisible(has)
        self.b_export.setEnabled(has)
        self.b_clear.setEnabled(has)
        self.model.set_frame(self.df, self.cols)
        self.table.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
        fit_columns(self.table)
        pill = self.model.cols.index(PRED_COL)
        self.table.setColumnWidth(pill, 130)

    def add(self, rec: dict):
        row = pd.DataFrame([{c: rec.get(c) for c in self.cols}])
        self.df = pd.concat([row, self.df], ignore_index=True).head(1000)
        self._save()
        self._refresh()

    def clear(self):
        ans = QtWidgets.QMessageBox.question(self, "Clear history", "Remove all saved analyses?")
        if ans == QtWidgets.QMessageBox.StandardButton.Yes:
            self.df = pd.DataFrame(columns=self.cols)
            self._save()
            self._refresh()

    def export(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export history", "aether_history.csv",
                                                        "CSV files (*.csv)")
        if path:
            self.df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Page: about the model
# ---------------------------------------------------------------------------
class AboutPage(QtWidgets.QScrollArea):
    def __init__(self, engine: Engine):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        inner = QtWidgets.QWidget()
        inner.setObjectName("page")
        self.setWidget(inner)
        root = QtWidgets.QVBoxLayout(inner)
        root.setContentsMargins(44, 36, 44, 36)
        root.setSpacing(22)
        root.addWidget(page_header("About the model", "How it decides.",
                                   "A short, honest description of what this classifier is and isn\u2019t."))

        # stats
        stats = QtWidgets.QHBoxLayout()
        stats.setSpacing(18)
        for big, cap in ((f"{MODEL_CARD['test_accuracy'] * 100:.1f}%", "Test accuracy"),
                         (f"{MODEL_CARD['test_macro_f1']:.3f}", "Test macro F1"),
                         (str(len(engine.order)), "AQI categories"),
                         (str(len(engine.cities)), "Cities supported")):
            c = Card()
            cv = QtWidgets.QVBoxLayout(c)
            cv.setContentsMargins(24, 18, 24, 18)
            cv.setSpacing(0)
            cv.addWidget(label(big, family=FONT_DISPLAY, px=44, weight=W.Medium))
            cv.addWidget(label(cap, px=10, weight=W.Medium, color=INK_SOFT, spacing=1.4, upper=True))
            stats.addWidget(c, 1)
        root.addLayout(stats)

        # how it works
        c = Card()
        cv = QtWidgets.QVBoxLayout(c)
        cv.setContentsMargins(30, 24, 30, 26)
        cv.setSpacing(10)
        cv.addWidget(label("The model", family=FONT_DISPLAY, px=19, weight=W.DemiBold))
        cv.addWidget(label(
            f"{MODEL_CARD['name']}. A gradient-boosted tree ensemble trained on {MODEL_CARD['train_data']}. "
            "Rare high-pollution classes were oversampled with SMOTE (training data only), hyper-parameters "
            "were tuned with cross-validated search, and output probabilities were isotonic-calibrated so the "
            "confidence shown is closer to how often such predictions are right.",
            px=13, weight=W.Light, wrap=True))
        cv.addWidget(label(
            "Inputs: eight pollutants (" + ", ".join(engine.pollutants) + "), city and month. "
            "Season is derived from the month.", px=13, weight=W.Light, color=INK_SOFT, wrap=True))
        root.addWidget(c)

        # legend
        c = Card()
        cv = QtWidgets.QVBoxLayout(c)
        cv.setContentsMargins(30, 24, 30, 24)
        cv.setSpacing(0)
        cv.addWidget(label("Categories", family=FONT_DISPLAY, px=19, weight=W.DemiBold))
        cv.addSpacing(4)
        cv.addWidget(label("Ranges and health statements follow India\u2019s CPCB AQI scheme.",
                           px=11, weight=W.Light, color=INK_SOFT))
        cv.addSpacing(10)
        for cat in engine.order:
            cv.addWidget(hline())
            r = QtWidgets.QHBoxLayout()
            r.setContentsMargins(0, 12, 0, 12)
            r.setSpacing(14)
            r.addWidget(Dot(SEVERITY[cat], 12), 0, Qt.AlignmentFlag.AlignTop)
            n = label(cat, family=FONT_DISPLAY, px=22, weight=W.Medium)
            n.setFixedWidth(120)
            r.addWidget(n, 0, Qt.AlignmentFlag.AlignTop)
            rg = label(AQI_RANGE[cat], px=11, weight=W.Medium, color=INK_SOFT, spacing=0.8)
            rg.setFixedWidth(110)
            r.addWidget(rg, 0, Qt.AlignmentFlag.AlignTop)
            r.addWidget(label(AQI_HEALTH[cat], px=12, weight=W.Light, wrap=True), 1)
            cv.addLayout(r)
        root.addWidget(c)

        # Limitations and external evidence
        c = Card()
        cv = QtWidgets.QVBoxLayout(c)
        cv.setContentsMargins(30, 24, 30, 26)
        cv.setSpacing(10)
        cv.addWidget(label("Limitations", family=FONT_DISPLAY, px=19, weight=W.DemiBold))
        for text_line in (
            "Trained on 20 Indian cities from 2015–2020. Held-out test accuracy was 82.7%, "
            "with a macro F1 score of 0.79.",
            "The model was trained where particulate matter drives almost every high-AQI day.",
            "This is a classifier, not a health advisory. For decisions that matter, use official monitoring.",
        ):
            cv.addWidget(label("—  " + text_line, px=13, weight=W.Light, wrap=True))
        root.addWidget(c)

        c = Card()
        cv = QtWidgets.QVBoxLayout(c)
        cv.setContentsMargins(30, 24, 30, 26)
        cv.setSpacing(10)
        cv.addWidget(label("External checks", family=FONT_DISPLAY, px=19, weight=W.DemiBold))
        for text_line in (
            "A related six-pollutant model scored about 83% on Pune (an unseen Indian city) "
            "and about 71% on Beijing (an unseen country).",
            "That related model was least reliable (about 56%) on days where ozone sets the AQI, "
            "compared with about 83% on other days.",
            "The model inside this app was not tested directly on external data.",
        ):
            cv.addWidget(label("—  " + text_line, px=13, weight=W.Light, wrap=True))
        root.addWidget(c)

        cv.addWidget(label("Supported cities", family=FONT_DISPLAY, px=19, weight=W.DemiBold))
        cv.addWidget(label("  \u00b7  ".join(engine.cities), px=13, weight=W.Light, color=INK_SOFT, wrap=True))
        root.addWidget(c)
        root.addStretch()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, engine: Engine):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} \u2014 Air Quality Classification")
        self.setMinimumSize(1180, 760)
        self.resize(1320, 840)
        self.setWindowIcon(self._icon())

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        h = QtWidgets.QHBoxLayout(central)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        side = QtWidgets.QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(232)
        sv = QtWidgets.QVBoxLayout(side)
        sv.setContentsMargins(0, 34, 0, 26)
        sv.setSpacing(0)
        sv.addWidget(label("aether", family=FONT_DISPLAY, px=25, weight=W.Bold, color=INK, spacing=-0.4),
                     0, Qt.AlignmentFlag.AlignLeft)
        sv.itemAt(0).widget().setContentsMargins(26, 0, 0, 0)
        cap = label("Air quality classifier", px=10, weight=W.Normal, color=INK_SOFT)
        cap.setContentsMargins(26, 4, 0, 0)
        sv.addWidget(cap)
        sv.addSpacing(44)

        self.stack = QtWidgets.QStackedWidget()
        self.single = SinglePage(engine)
        self.batch = BatchPage(engine)
        self.history = HistoryPage(engine)
        self.about = AboutPage(engine)
        self.single.analyzed.connect(self.history.add)
        group = QtWidgets.QButtonGroup(self)
        for i, (name, page) in enumerate((("Single reading", self.single), ("Batch analysis", self.batch),
                                          ("History", self.history), ("About the model", self.about))):
            self.stack.addWidget(page)
            b = QtWidgets.QPushButton(name)
            b.setObjectName("nav")
            b.setCheckable(True)
            b.setFont(mkfont(FONT_UI, 11, W.DemiBold))
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            group.addButton(b, i)
            sv.addWidget(b)
            QtGui.QShortcut(QtGui.QKeySequence(f"Ctrl+{i + 1}"), self, activated=lambda i=i: group.button(i).click())
        group.idClicked.connect(self.stack.setCurrentIndex)
        self.stack.currentChanged.connect(lambda i: group.button(i).setChecked(True))   # nav always follows the page
        group.button(0).setChecked(True)
        sv.addStretch()
        foot = label(f"Model ready\n{len(engine.cities)} cities \u00b7 {len(engine.order)} categories\nv{APP_VERSION}",
                     px=10, weight=W.Light, color=INK_SOFT, spacing=0.6)
        foot.setContentsMargins(26, 0, 0, 0)
        sv.addWidget(foot)

        h.addWidget(side)
        h.addWidget(self.stack, 1)

    @staticmethod
    def _icon() -> QtGui.QIcon:
        pm = QtGui.QPixmap(128, 128)
        pm.fill(Qt.GlobalColor.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setBrush(QtGui.QColor(ACCENT))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QtCore.QRectF(4, 4, 120, 120), 26, 26)
        p.setPen(QtGui.QColor(PAPER))
        p.setFont(mkfont(FONT_DISPLAY, 76, W.Bold))
        p.drawText(QtCore.QRectF(0, 0, 128, 118), Qt.AlignmentFlag.AlignCenter, "A")
        p.end()
        return QtGui.QIcon(pm)


def main() -> int:
    QtCore.QCoreApplication.setApplicationName(APP_NAME)
    QtCore.QCoreApplication.setOrganizationName(APP_NAME)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    load_fonts()
    app.setFont(mkfont(FONT_UI, 13))
    app.setStyleSheet(build_qss())

    try:
        engine = Engine(MODEL_DIR)
    except Exception as ex:
        QtWidgets.QMessageBox.critical(
            None, "Couldn\u2019t load the model",
            f"The model files could not be loaded from:\n{MODEL_DIR}\n\n"
            "Expected: xgb_aqi_model.pkl, label_encoder.pkl, feature_columns.pkl, "
            f"pollutant_columns.pkl, city_list.pkl, season_list.pkl\n\nDetails: {ex}")
        return 1

    win = MainWindow(engine)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
