"""
Air Quality Classification App (v2)
--------------------------------------
Adds:
  - A probability bar chart (all 6 classes) alongside the top prediction
  - General UI polish: padding, framed sections, consistent styling

Expects a `models/` folder (next to this file) containing the artifacts
saved from the notebook via joblib:
    xgb_aqi_model.pkl, label_encoder.pkl, feature_columns.pkl,
    pollutant_columns.pkl, city_list.pkl, season_list.pkl
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import joblib
import os
from datetime import datetime

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

MODEL_DIR = 'models'

# ---------------------------------------------------------------------------
# Load trained model + preprocessing artifacts
# ---------------------------------------------------------------------------
model = joblib.load(os.path.join(MODEL_DIR, 'xgb_aqi_model.pkl'))
le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
feature_columns = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))
pollutant_cols = joblib.load(os.path.join(MODEL_DIR, 'pollutant_columns.pkl'))
city_list = joblib.load(os.path.join(MODEL_DIR, 'city_list.pkl'))
season_list = joblib.load(os.path.join(MODEL_DIR, 'season_list.pkl'))

# Fixed left-to-right severity order for the chart, regardless of alphabetical
# order from LabelEncoder — this reads more naturally to a viewer.
AQI_ORDER = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
AQI_ORDER = [c for c in AQI_ORDER if c in le.classes_]  # keep only classes actually present

AQI_COLORS = {
    'Good': '#2ecc71',
    'Satisfactory': '#a6d96a',
    'Moderate': '#f1c40f',
    'Poor': '#e67e22',
    'Very Poor': '#e74c3c',
    'Severe': '#7d0f0f',
}

BG_COLOR = '#f4f6f8'
CARD_COLOR = '#ffffff'
ACCENT_COLOR = '#2c3e50'


# ---------------------------------------------------------------------------
# Preprocessing helpers — must mirror exactly what was done in the notebook
# ---------------------------------------------------------------------------
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Summer'
    elif month in [6, 7, 8, 9]:
        return 'Monsoon'
    else:
        return 'Post-Monsoon'


def build_feature_row(pollutant_values, city, month):
    season = get_season(month)
    row = {col: pollutant_values.get(col, 0.0) for col in pollutant_cols}
    row['Month'] = month
    row['City'] = city
    row['Season'] = season

    df_row = pd.DataFrame([row])
    df_encoded = pd.get_dummies(df_row, columns=['City', 'Season'])
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)
    return df_encoded


def predict_single(pollutant_values, city, month):
    """Returns (predicted_label, {class_label: probability, ...})"""
    X = build_feature_row(pollutant_values, city, month)
    proba = model.predict_proba(X)[0]
    pred_enc = proba.argmax()
    pred_label = le.inverse_transform([pred_enc])[0]

    proba_dict = {le.inverse_transform([i])[0]: p for i, p in enumerate(proba)}
    return pred_label, proba_dict


def predict_batch(df_input):
    df_input = df_input.copy()

    if 'Date' in df_input.columns:
        df_input['Date'] = pd.to_datetime(df_input['Date'])
        df_input['Month'] = df_input['Date'].dt.month
    elif 'Month' not in df_input.columns:
        raise ValueError("CSV must contain either a 'Date' or a 'Month' column")

    if 'City' not in df_input.columns:
        raise ValueError("CSV must contain a 'City' column")

    missing_pollutants = [c for c in pollutant_cols if c not in df_input.columns]
    if missing_pollutants:
        raise ValueError(f"CSV is missing required pollutant columns: {missing_pollutants}")

    df_input['Season'] = df_input['Month'].apply(get_season)

    df_encoded = pd.get_dummies(df_input, columns=['City', 'Season'])
    X = df_encoded.reindex(columns=feature_columns, fill_value=0)

    preds_enc = model.predict(X)
    preds = le.inverse_transform(preds_enc)
    df_input['Predicted_AQI_Bucket'] = preds
    return df_input


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class AQIApp:
    def __init__(self, root):
        self.root = root
        root.title("Air Quality Classification App")
        root.geometry("980x700")
        root.configure(bg=BG_COLOR)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background=BG_COLOR)
        style.configure('TNotebook.Tab', font=('Segoe UI', 10, 'bold'), padding=[14, 8])
        style.configure('TFrame', background=BG_COLOR)
        style.configure('Card.TFrame', background=CARD_COLOR)
        style.configure('TLabel', background=BG_COLOR, font=('Segoe UI', 10))
        style.configure('Card.TLabel', background=CARD_COLOR, font=('Segoe UI', 10))
        style.configure('Header.TLabel', background=BG_COLOR, font=('Segoe UI', 15, 'bold'), foreground=ACCENT_COLOR)
        style.configure('TButton', font=('Segoe UI', 10, 'bold'), padding=8)
        style.configure('TEntry', padding=4)
        style.configure('TCombobox', padding=4)

        notebook = ttk.Notebook(root)
        notebook.pack(fill='both', expand=True, padx=12, pady=12)

        self.manual_tab = ttk.Frame(notebook)
        self.batch_tab = ttk.Frame(notebook)
        notebook.add(self.manual_tab, text="  Manual Entry  ")
        notebook.add(self.batch_tab, text="  Batch CSV Prediction  ")

        self.build_manual_tab()
        self.build_batch_tab()

    # -- Manual entry tab ---------------------------------------------------
    def build_manual_tab(self):
        outer = ttk.Frame(self.manual_tab, padding=15)
        outer.pack(fill='both', expand=True)

        # Left card: inputs
        left = ttk.Frame(outer, style='Card.TFrame', padding=20)
        left.grid(row=0, column=0, sticky='n', padx=(0, 15))

        ttk.Label(left, text="Pollutant Readings", style='Header.TLabel', background=CARD_COLOR).grid(
            row=0, column=0, columnspan=2, pady=(0, 15), sticky='w')

        self.entries = {}
        for i, col in enumerate(pollutant_cols):
            ttk.Label(left, text=f"{col} (\u00b5g/m\u00b3)", style='Card.TLabel').grid(
                row=i + 1, column=0, sticky='w', pady=5)
            entry = ttk.Entry(left, width=14)
            entry.grid(row=i + 1, column=1, pady=5, padx=(10, 0))
            self.entries[col] = entry

        row_offset = len(pollutant_cols) + 1

        ttk.Label(left, text="City", style='Card.TLabel').grid(row=row_offset, column=0, sticky='w', pady=(15, 5))
        self.city_var = tk.StringVar(value=city_list[0])
        ttk.Combobox(left, textvariable=self.city_var, values=city_list,
                     state='readonly', width=14).grid(row=row_offset, column=1, pady=(15, 5), padx=(10, 0))

        ttk.Label(left, text="Month", style='Card.TLabel').grid(row=row_offset + 1, column=0, sticky='w', pady=5)
        self.month_var = tk.StringVar(value=str(datetime.now().month))
        ttk.Combobox(left, textvariable=self.month_var, values=[str(m) for m in range(1, 13)],
                     state='readonly', width=14).grid(row=row_offset + 1, column=1, pady=5, padx=(10, 0))

        ttk.Button(left, text="Predict AQI Category", command=self.on_predict_manual).grid(
            row=row_offset + 2, column=0, columnspan=2, pady=(20, 0), sticky='ew')

        # Right card: result + chart
        right = ttk.Frame(outer, style='Card.TFrame', padding=20)
        right.grid(row=0, column=1, sticky='n')

        ttk.Label(right, text="Prediction", style='Header.TLabel', background=CARD_COLOR).pack(
            anchor='w', pady=(0, 10))

        self.result_label = tk.Label(right, text="Enter readings and click Predict",
                                      font=('Segoe UI', 18, 'bold'), bg=CARD_COLOR, fg='#7f8c8d')
        self.result_label.pack(anchor='w', pady=(0, 15))

        self.fig, self.ax = plt.subplots(figsize=(5, 3.2), dpi=100)
        self.fig.patch.set_facecolor(CARD_COLOR)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack()
        self.draw_empty_chart()

    def draw_empty_chart(self):
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_yticks(range(len(AQI_ORDER)))
        self.ax.set_yticklabels(AQI_ORDER)
        self.ax.invert_yaxis()
        self.ax.set_xlabel('Probability')
        self.ax.set_facecolor(CARD_COLOR)
        self.fig.tight_layout()
        self.canvas.draw()

    def on_predict_manual(self):
        try:
            pollutant_values = {}
            for col, entry in self.entries.items():
                val = entry.get().strip()
                pollutant_values[col] = float(val) if val else 0.0

            city = self.city_var.get()
            month = int(self.month_var.get())

            pred_label, proba_dict = predict_single(pollutant_values, city, month)
            confidence = proba_dict[pred_label]

            color = AQI_COLORS.get(pred_label, ACCENT_COLOR)
            self.result_label.config(
                text=f"{pred_label}  ({confidence * 100:.1f}% confidence)", fg=color)

            # Draw probability bars for every class, in fixed severity order
            probs = [proba_dict.get(cls, 0.0) for cls in AQI_ORDER]
            colors = [AQI_COLORS.get(cls, '#95a5a6') for cls in AQI_ORDER]

            self.ax.clear()
            bars = self.ax.barh(AQI_ORDER, probs, color=colors)
            self.ax.set_xlim(0, 1)
            self.ax.invert_yaxis()
            self.ax.set_xlabel('Probability')
            self.ax.set_facecolor(CARD_COLOR)
            for bar, p in zip(bars, probs):
                self.ax.text(min(p + 0.02, 0.93), bar.get_y() + bar.get_height() / 2,
                             f"{p*100:.0f}%", va='center', fontsize=9)
            self.fig.tight_layout()
            self.canvas.draw()

        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values for all pollutant fields.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # -- Batch CSV tab --------------------------------------------------------
    def build_batch_tab(self):
        outer = ttk.Frame(self.batch_tab, padding=20)
        outer.pack(fill='both', expand=True)

        ttk.Label(outer, text="Batch Prediction from CSV", style='Header.TLabel').pack(anchor='w', pady=(0, 5))
        ttk.Label(outer, text="Required columns: City, Date (or Month), " + ", ".join(pollutant_cols),
                  wraplength=800, justify='left').pack(anchor='w', pady=(0, 15))

        btn_frame = ttk.Frame(outer)
        btn_frame.pack(anchor='w', pady=5)
        ttk.Button(btn_frame, text="Upload CSV", command=self.on_upload_csv).pack(side='left', padx=(0, 8))
        self.export_btn = ttk.Button(btn_frame, text="Export Results", command=self.on_export_csv, state='disabled')
        self.export_btn.pack(side='left')

        columns = ['City', 'Date', 'PM2.5', 'PM10', 'Predicted_AQI_Bucket']
        self.tree = ttk.Treeview(outer, columns=columns, show='headings', height=16)
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=140, anchor='center')
        self.tree.pack(fill='both', expand=True, pady=(15, 0))

        self.batch_result_df = None

    def on_upload_csv(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not filepath:
            return
        try:
            df_input = pd.read_csv(filepath)
            result_df = predict_batch(df_input)
            self.batch_result_df = result_df
            self.populate_tree(result_df)
            self.export_btn.config(state='normal')
            messagebox.showinfo("Success", f"Predicted AQI category for {len(result_df)} rows.")
        except Exception as e:
            messagebox.showerror("Error processing CSV", str(e))

    def populate_tree(self, df_result):
        for row in self.tree.get_children():
            self.tree.delete(row)

        display_cols = ['City']
        if 'Date' in df_result.columns:
            display_cols.append('Date')
        display_cols += ['PM2.5', 'PM10', 'Predicted_AQI_Bucket']

        for _, row in df_result.iterrows():
            values = [row.get(c, '') for c in display_cols]
            self.tree.insert('', 'end', values=values)

    def on_export_csv(self):
        if self.batch_result_df is None:
            return
        filepath = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[("CSV files", "*.csv")])
        if filepath:
            self.batch_result_df.to_csv(filepath, index=False)
            messagebox.showinfo("Saved", f"Results saved to {filepath}")


if __name__ == '__main__':
    root = tk.Tk()
    app = AQIApp(root)
    root.mainloop()