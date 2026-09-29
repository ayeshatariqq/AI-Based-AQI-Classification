"""
Air Quality Classification App (v3)
--------------------------------------
Adds, on top of v2:
  - Card-styled batch prediction tab matching the Manual Entry tab's look
  - Color-coded result rows in the batch table (by predicted AQI category)
  - A summary bar chart showing the distribution of predicted categories
  - A colored result "chip" behind the manual-entry prediction
  - Clear/reset buttons on both tabs

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
from collections import Counter

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

AQI_ORDER = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
AQI_ORDER = [c for c in AQI_ORDER if c in le.classes_]

AQI_COLORS = {
    'Good': '#2ecc71',
    'Satisfactory': '#a6d96a',
    'Moderate': '#f1c40f',
    'Poor': '#e67e22',
    'Very Poor': '#e74c3c',
    'Severe': '#7d0f0f',
}

# Lighter tint of each color, used as a row background in the results table
# so the text stays readable while still signalling severity.
AQI_ROW_TINTS = {
    'Good': '#eafaf1',
    'Satisfactory': '#f2f9e8',
    'Moderate': '#fef9e7',
    'Poor': '#fdf0e2',
    'Very Poor': '#fdecea',
    'Severe': '#f6dede',
}

BG_COLOR = '#f4f6f8'
CARD_COLOR = '#ffffff'
ACCENT_COLOR = '#2c3e50'
MUTED_TEXT = '#7f8c8d'


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
        root.geometry("1040x740")
        root.configure(bg=BG_COLOR)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background=BG_COLOR)
        style.configure('TNotebook.Tab', font=('Segoe UI', 10, 'bold'), padding=[14, 8])
        style.configure('TFrame', background=BG_COLOR)
        style.configure('Card.TFrame', background=CARD_COLOR)
        style.configure('TLabel', background=BG_COLOR, font=('Segoe UI', 10))
        style.configure('Card.TLabel', background=CARD_COLOR, font=('Segoe UI', 10))
        style.configure('Muted.Card.TLabel', background=CARD_COLOR, font=('Segoe UI', 9), foreground=MUTED_TEXT)
        style.configure('Header.TLabel', background=BG_COLOR, font=('Segoe UI', 15, 'bold'), foreground=ACCENT_COLOR)
        style.configure('Header.Card.TLabel', background=CARD_COLOR, font=('Segoe UI', 14, 'bold'), foreground=ACCENT_COLOR)
        style.configure('TButton', font=('Segoe UI', 10, 'bold'), padding=8)
        style.configure('Secondary.TButton', font=('Segoe UI', 9))
        style.configure('TEntry', padding=4)
        style.configure('TCombobox', padding=4)

        style.configure('Results.Treeview', font=('Segoe UI', 9), rowheight=26)
        style.configure('Results.Treeview.Heading', font=('Segoe UI', 9, 'bold'))

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

        left = ttk.Frame(outer, style='Card.TFrame', padding=20)
        left.grid(row=0, column=0, sticky='n', padx=(0, 15))

        ttk.Label(left, text="Pollutant Readings", style='Header.Card.TLabel').grid(
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

        btn_row = ttk.Frame(left, style='Card.TFrame')
        btn_row.grid(row=row_offset + 2, column=0, columnspan=2, pady=(20, 0), sticky='ew')
        ttk.Button(btn_row, text="Predict AQI Category", command=self.on_predict_manual).pack(
            side='left', fill='x', expand=True)
        ttk.Button(btn_row, text="Clear", style='Secondary.TButton', command=self.on_clear_manual).pack(
            side='left', padx=(8, 0))

        # Right card: result chip + chart
        right = ttk.Frame(outer, style='Card.TFrame', padding=20)
        right.grid(row=0, column=1, sticky='n')

        ttk.Label(right, text="Prediction", style='Header.Card.TLabel').pack(anchor='w', pady=(0, 10))

        self.result_chip = tk.Frame(right, bg='#ecf0f1', height=56)
        self.result_chip.pack(fill='x', pady=(0, 15))
        self.result_chip.pack_propagate(False)
        self.result_label = tk.Label(self.result_chip, text="Enter readings and click Predict",
                                      font=('Segoe UI', 15, 'bold'), bg='#ecf0f1', fg=MUTED_TEXT)
        self.result_label.pack(expand=True, fill='both')

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

    def on_clear_manual(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self.result_chip.configure(bg='#ecf0f1')
        self.result_label.configure(text="Enter readings and click Predict", bg='#ecf0f1', fg=MUTED_TEXT)
        self.draw_empty_chart()

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
            self.result_chip.configure(bg=color)
            self.result_label.configure(
                text=f"{pred_label}   \u2014   {confidence * 100:.1f}% confidence",
                bg=color, fg='#ffffff')

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
        outer = ttk.Frame(self.batch_tab, padding=15)
        outer.pack(fill='both', expand=True)

        # Top card: upload controls + requirements
        top_card = ttk.Frame(outer, style='Card.TFrame', padding=20)
        top_card.pack(fill='x', pady=(0, 15))

        ttk.Label(top_card, text="Batch Prediction from CSV", style='Header.Card.TLabel').pack(anchor='w')
        ttk.Label(top_card, text="Required columns: City, Date (or Month), " + ", ".join(pollutant_cols),
                  style='Muted.Card.TLabel', wraplength=900, justify='left').pack(anchor='w', pady=(4, 15))

        btn_frame = ttk.Frame(top_card, style='Card.TFrame')
        btn_frame.pack(anchor='w')
        ttk.Button(btn_frame, text="Upload CSV", command=self.on_upload_csv).pack(side='left', padx=(0, 8))
        self.export_btn = ttk.Button(btn_frame, text="Export Results", command=self.on_export_csv, state='disabled')
        self.export_btn.pack(side='left', padx=(0, 8))
        self.clear_btn = ttk.Button(btn_frame, text="Clear", style='Secondary.TButton',
                                     command=self.on_clear_batch, state='disabled')
        self.clear_btn.pack(side='left')

        # Middle: results table (card) + summary chart (card), side by side
        middle = ttk.Frame(outer)
        middle.pack(fill='both', expand=True)

        table_card = ttk.Frame(middle, style='Card.TFrame', padding=15)
        table_card.pack(side='left', fill='both', expand=True, padx=(0, 15))

        ttk.Label(table_card, text="Predictions", style='Card.TLabel', font=('Segoe UI', 11, 'bold')).pack(
            anchor='w', pady=(0, 8))

        columns = ['City', 'Date', 'PM2.5', 'PM10', 'Predicted_AQI_Bucket']
        self.tree = ttk.Treeview(table_card, columns=columns, show='headings', height=16, style='Results.Treeview')
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor='center')
        self.tree.pack(fill='both', expand=True)

        for cls, tint in AQI_ROW_TINTS.items():
            self.tree.tag_configure(cls, background=tint)

        # Summary card: class distribution chart
        summary_card = ttk.Frame(middle, style='Card.TFrame', padding=15, width=280)
        summary_card.pack(side='left', fill='y')
        summary_card.pack_propagate(False)

        ttk.Label(summary_card, text="Category Breakdown", style='Card.TLabel', font=('Segoe UI', 11, 'bold')).pack(
            anchor='w', pady=(0, 8))

        self.summary_fig, self.summary_ax = plt.subplots(figsize=(3.0, 4.2), dpi=100)
        self.summary_fig.patch.set_facecolor(CARD_COLOR)
        self.summary_canvas = FigureCanvasTkAgg(self.summary_fig, master=summary_card)
        self.summary_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.draw_empty_summary()

        self.summary_text = ttk.Label(summary_card, text="Upload a CSV to see results",
                                       style='Muted.Card.TLabel', wraplength=240, justify='left')
        self.summary_text.pack(anchor='w', pady=(8, 0))

        self.batch_result_df = None

    def draw_empty_summary(self):
        self.summary_ax.clear()
        self.summary_ax.axis('off')
        self.summary_fig.tight_layout()
        self.summary_canvas.draw()

    def on_upload_csv(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not filepath:
            return
        try:
            df_input = pd.read_csv(filepath)
            result_df = predict_batch(df_input)
            self.batch_result_df = result_df
            self.populate_tree(result_df)
            self.draw_summary(result_df)
            self.export_btn.config(state='normal')
            self.clear_btn.config(state='normal')
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
            cls = row.get('Predicted_AQI_Bucket', '')
            self.tree.insert('', 'end', values=values, tags=(cls,))

    def draw_summary(self, df_result):
        counts = Counter(df_result['Predicted_AQI_Bucket'])
        labels = [c for c in AQI_ORDER if counts.get(c, 0) > 0]
        values = [counts[c] for c in labels]
        colors = [AQI_COLORS.get(c, '#95a5a6') for c in labels]

        self.summary_ax.clear()
        self.summary_ax.axis('on')
        bars = self.summary_ax.barh(labels, values, color=colors)
        self.summary_ax.invert_yaxis()
        self.summary_ax.set_xlabel('Count')
        self.summary_ax.set_facecolor(CARD_COLOR)
        for bar, v in zip(bars, values):
            self.summary_ax.text(v + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                                  str(v), va='center', fontsize=8)
        self.summary_fig.tight_layout()
        self.summary_canvas.draw()

        total = len(df_result)
        top_class = max(counts, key=counts.get)
        self.summary_text.configure(
            text=f"{total} rows classified.\nMost common: {top_class} ({counts[top_class]} rows, "
                 f"{counts[top_class]/total*100:.0f}%)")

    def on_clear_batch(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        self.batch_result_df = None
        self.export_btn.config(state='disabled')
        self.clear_btn.config(state='disabled')
        self.draw_empty_summary()
        self.summary_text.configure(text="Upload a CSV to see results")

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