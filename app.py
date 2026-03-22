import os

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
from pathlib import Path
from datetime import datetime

from chatbot import data, init_session_metrics, process_user_query


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=DM+Serif+Display:ital@0;1&display=swap');

:root {
  --bg1: #f8f4ea;
  --bg2: #e6f0ff;
  --card: #ffffff;
  --ink: #1e2430;
  --muted: #4d596a;
  --accent: #0e7a6d;
  --line: #d4dbe7;
}

:root[data-theme='dark'] {
  --bg1: #0f1722;
  --bg2: #111a2e;
  --card: #172233;
  --ink: #e7eef8;
  --muted: #a8b6c9;
  --accent: #39bfa8;
  --line: #2c3f57;
}

.gradio-container {
  font-family: 'Space Grotesk', sans-serif !important;
  background: radial-gradient(circle at 15% 10%, #fff7e8 0%, var(--bg1) 35%, var(--bg2) 100%) !important;
  min-height: 100vh;
  color: var(--ink) !important;
  transition: background 0.45s ease, color 0.3s ease;
}

#main-card {
  max-width: 1120px;
  margin: 16px auto;
  border: 1px solid var(--line);
  border-radius: 22px;
  background: linear-gradient(180deg, #ffffff 0%, #fdfefe 100%);
  box-shadow: 0 14px 45px rgba(18, 36, 66, 0.12);
  overflow: hidden;
}

#hero {
  background: linear-gradient(120deg, #0e7a6d 0%, #2e9ec0 55%, #d66f3d 100%);
  color: #fff;
  padding: 20px 24px;
}

#hero h1 {
  margin: 0;
  font-family: 'DM Serif Display', serif;
  font-size: 34px;
}

#hero p {
  margin: 8px 0 0;
  font-size: 14px;
}

#panel {
  padding: 18px;
}

#panel * {
  transition: background-color 0.3s ease, border-color 0.3s ease, color 0.3s ease;
}

#hint {
  color: var(--muted);
  font-size: 13px;
  margin-bottom: 8px;
}

#quick button {
  border-radius: 999px !important;
  border: 1px solid #c8d7f2 !important;
  background: #f0f6ff !important;
  color: #22426f !important;
}

#category-chips label {
  border: 1px solid var(--line) !important;
  border-radius: 999px !important;
  padding: 2px 10px !important;
  background: #eef5ff !important;
}

#category-chips label span {
  color: #1f2b3d !important;
  font-weight: 600 !important;
}

:root[data-theme='dark'] #category-chips label {
  background: #1b2a3e !important;
  border-color: #3a5270 !important;
}

:root[data-theme='dark'] #category-chips label span {
  color: #edf4ff !important;
}

#send-btn {
  background: linear-gradient(120deg, var(--accent) 0%, #16a085 100%) !important;
  color: #fff !important;
  border: none !important;
}

#pdf-btn {
  background: linear-gradient(120deg, #d66f3d 0%, #ef955b 100%) !important;
  color: #fff !important;
  border: none !important;
}

#assistant-box {
  border: 1px solid var(--line);
  border-radius: 14px;
  background: #fcfdff;
}

:root[data-theme='dark'] #assistant-box {
  background: #111b2c;
}

footer { visibility: hidden; }
"""


WELCOME_MSG = (
  "Welcome to your interactive medical assistant.\n\n"
  "Commands: `metrics`, `options`, `name: <new name>`.\n"
  "Tip: Ask naturally, e.g. `My stomach burns after spicy food`."
)


def _format_bot_text(text):
  if "PRESCRIPTION STYLE HEALTH REPORT" in text:
    return f"```text\n{text}\n```"
  return text


def _extract_report_text(history):
  if not history:
    return ""
  for msg in reversed(history):
    if isinstance(msg, dict):
      if msg.get("role") != "assistant":
        continue
      content = msg.get("content", "")
    elif isinstance(msg, (list, tuple)) and len(msg) == 2:
      # Gradio 3 Chatbot format: [user_message, bot_message]
      content = msg[1] or ""
    else:
      continue
    if "PRESCRIPTION STYLE HEALTH REPORT" in content:
      if content.startswith("```text"):
        content = content.replace("```text", "", 1).rstrip("`").strip()
      return content
  return ""


def _write_basic_pdf(text, output_path):
  lines = [line[:110] for line in text.splitlines()]
  lines = lines[:90] if lines else ["No prescription generated yet."]

  def esc(s):
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

  content_lines = ["BT", "/F1 10 Tf", "50 790 Td", "12 TL"]
  for idx, line in enumerate(lines):
    if idx == 0:
      content_lines.append(f"({esc(line)}) Tj")
    else:
      content_lines.append("T*")
      content_lines.append(f"({esc(line)}) Tj")
  content_lines.append("ET")
  stream = "\n".join(content_lines).encode("latin-1", errors="replace")

  objects = []
  objects.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
  objects.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
  objects.append(
    b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n"
  )
  objects.append(b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Courier >> endobj\n")
  objects.append(
    f"5 0 obj << /Length {len(stream)} >> stream\n".encode("latin-1") + stream + b"\nendstream endobj\n"
  )

  pdf = bytearray(b"%PDF-1.4\n")
  xref_positions = [0]
  for obj in objects:
    xref_positions.append(len(pdf))
    pdf.extend(obj)

  xref_start = len(pdf)
  pdf.extend(f"xref\n0 {len(xref_positions)}\n".encode("latin-1"))
  pdf.extend(b"0000000000 65535 f \n")
  for pos in xref_positions[1:]:
    pdf.extend(f"{pos:010d} 00000 n \n".encode("latin-1"))
  pdf.extend((f"trailer << /Size {len(xref_positions)} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF").encode("latin-1"))

  Path(output_path).write_bytes(pdf)


def export_prescription_pdf(history, patient_name):
  report_text = _extract_report_text(history)
  if not report_text:
    report_text = "No prescription report available yet. Generate one and try again."

  safe_name = "".join(ch for ch in patient_name if ch.isalnum() or ch in ("_", "-")) or "Patient"
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  out_dir = Path("generated_reports")
  out_dir.mkdir(exist_ok=True)
  output_path = out_dir / f"prescription_{safe_name}_{timestamp}.pdf"
  _write_basic_pdf(report_text, output_path)
  return str(output_path)


def get_category_choices():
  return sorted(set(str(c).strip() for c in data["Category"].dropna().tolist() if str(c).strip()))


def get_conditions_for_categories(selected_categories):
  df = data.copy()
  if selected_categories:
    df = df[df["Category"].isin(selected_categories)]

  conditions = set()
  for q in df["Question"].dropna().tolist():
    question = str(q).strip().lower()
    if question.startswith("what is "):
      conditions.add(str(q)[8:].strip().title())
    elif question.startswith("what causes "):
      conditions.add(str(q)[12:].strip().title())
    elif question.startswith("symptoms of "):
      conditions.add(str(q)[12:].strip().title())
  return sorted(conditions)


def update_condition_dropdown(selected_categories):
  choices = get_conditions_for_categories(selected_categories)
  return gr.update(choices=choices, value=choices[0] if choices else None)


def use_selected_condition(condition_name):
  return f"What is {condition_name}" if condition_name else ""


def submit_query(user_message, patient_name, history, cache_state, metrics_state):
  user_message = (user_message or "").strip()
  patient_name = (patient_name or "Patient").strip() or "Patient"

  if history is None:
    history = [[None, WELCOME_MSG]]
  elif isinstance(history, tuple):
    history = list(history)

  if not isinstance(cache_state, dict):
    cache_state = {}

  required_metric_keys = {
    "total_queries",
    "answered",
    "unanswered",
    "cache_hits",
    "cache_misses",
    "response_times_ms",
    "confidence_sum",
    "confidence_count",
    "evidence_total",
    "evidence_possible",
    "safety_reports",
    "method_counter",
    "condition_counter",
  }
  if not isinstance(metrics_state, dict) or not required_metric_keys.issubset(set(metrics_state.keys())):
    metrics_state = init_session_metrics()

  if not user_message:
    return "", patient_name, history, cache_state, metrics_state

  try:
    result = process_user_query(user_message, patient_name, cache_state, metrics_state)
    patient_name = result.get("patient_name", patient_name)
    bot_message = _format_bot_text(result.get("message", "No response generated."))
  except Exception:
    bot_message = "Processing issue handled. Displaying best available result."

  history = history + [[user_message, bot_message]]
  return "", patient_name, history, cache_state, metrics_state


def quick_fill(example):
  return example


def clear_chat(patient_name, cache_state, metrics_state):
  metrics_state = init_session_metrics()
  cache_state = {}
  history = [[None, WELCOME_MSG]]
  return history, patient_name, cache_state, metrics_state


def set_theme_mode(mode):
  mode = (mode or "Light").strip().lower()
  return "dark" if mode == "dark" else "light"


def build_ui():
  with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
    cache_state = gr.State({})
    metrics_state = gr.State(init_session_metrics())
    theme_state = gr.State("light")

    with gr.Column(elem_id="main-card"):
      gr.HTML(
        """
        <div id='hero'>
          <h1>Medical Assistant UI</h1>
          <p>Prescription-style, evidence-backed, and interactive experience.</p>
        </div>
        """
      )

      with gr.Column(elem_id="panel"):
        theme_toggle = gr.Radio(
          choices=["Light", "Dark"],
          value="Light",
          label="Theme",
        )

        patient_name = gr.Textbox(label="Patient Name", value="Patient", placeholder="Enter patient name")

        gr.Markdown("<div id='hint'>Ask about causes, symptoms, definitions, or describe symptoms in plain language.</div>")

        chatbot = gr.Chatbot(
          value=[[None, WELCOME_MSG]],
          elem_id="assistant-box",
          height=480,
        )

        with gr.Row():
          user_input = gr.Textbox(label="Your Query", placeholder="Type your medical question...", scale=5)
          send_btn = gr.Button("Generate Prescription", elem_id="send-btn", scale=1)

        category_chips = gr.CheckboxGroup(
          choices=get_category_choices(),
          label="Category Filters",
          value=[],
          elem_id="category-chips",
        )

        with gr.Row():
          condition_picker = gr.Dropdown(
            choices=get_conditions_for_categories([]),
            label="Condition Picker",
            value=None,
            scale=4,
          )
          use_condition_btn = gr.Button("Use Condition", scale=1)

        with gr.Row(elem_id="quick"):
          q1 = gr.Button("What causes malaria?")
          q2 = gr.Button("Symptoms of ringworm")
          q3 = gr.Button("My stomach burns after spicy food")
          q4 = gr.Button("metrics")
          q5 = gr.Button("options")

        clear_btn = gr.Button("Reset Session")
        pdf_btn = gr.Button("Download Last Prescription PDF", elem_id="pdf-btn")
        pdf_file = gr.File(label="Generated PDF", interactive=False)

        send_btn.click(
          submit_query,
          inputs=[user_input, patient_name, chatbot, cache_state, metrics_state],
          outputs=[user_input, patient_name, chatbot, cache_state, metrics_state],
        )
        user_input.submit(
          submit_query,
          inputs=[user_input, patient_name, chatbot, cache_state, metrics_state],
          outputs=[user_input, patient_name, chatbot, cache_state, metrics_state],
        )

        q1.click(quick_fill, inputs=[gr.State("What causes malaria?")], outputs=[user_input])
        q2.click(quick_fill, inputs=[gr.State("Symptoms of ringworm")], outputs=[user_input])
        q3.click(quick_fill, inputs=[gr.State("My stomach burns after spicy food")], outputs=[user_input])
        q4.click(quick_fill, inputs=[gr.State("metrics")], outputs=[user_input])
        q5.click(quick_fill, inputs=[gr.State("options")], outputs=[user_input])

        category_chips.change(update_condition_dropdown, inputs=[category_chips], outputs=[condition_picker])
        use_condition_btn.click(use_selected_condition, inputs=[condition_picker], outputs=[user_input])
        pdf_btn.click(export_prescription_pdf, inputs=[chatbot, patient_name], outputs=[pdf_file])

        theme_toggle.change(
          set_theme_mode,
          inputs=[theme_toggle],
          outputs=[theme_state],
          js="(mode)=>{const val=(mode||'Light').toLowerCase(); document.documentElement.setAttribute('data-theme', val==='dark'?'dark':'light'); return val==='dark'?'dark':'light';}",
        )

        clear_btn.click(
          clear_chat,
          inputs=[patient_name, cache_state, metrics_state],
          outputs=[chatbot, patient_name, cache_state, metrics_state],
        )

  return demo


if __name__ == "__main__":
  app = build_ui()
  launch_info = app.launch(share=True)
  if isinstance(launch_info, tuple) and len(launch_info) >= 3:
    share_url = launch_info[2]
    if share_url:
      print(f"Public URL: {share_url}")
