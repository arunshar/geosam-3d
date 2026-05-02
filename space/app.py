"""Gradio HF Space for GeoSAM-3D: upload video, click frame, get 3D mask."""
from __future__ import annotations

import gradio as gr


def segment(video_file, click_frame_idx: int, click_xy: str):
    """Run the full pipeline: MonoGS -> features -> heat-method propagation."""
    return None, "[Stub: this would render an interactive 3DGS mask in WebGL]"


def build_ui():
    with gr.Blocks(title="GeoSAM-3D") as demo:
        gr.Markdown("# GeoSAM-3D\nUpload a short monocular video, click any frame, get a 3D mask.")
        with gr.Row():
            video = gr.Video(label="Monocular video (~30 frames)")
            with gr.Column():
                click_frame = gr.Slider(0, 29, value=15, step=1, label="Frame to prompt")
                click_xy = gr.Textbox(label="Click point as 'x,y'", value="320,240")
                run = gr.Button("Segment in 3D", variant="primary")
        with gr.Row():
            out = gr.Image(label="3D mask preview")
            stats = gr.Textbox(label="Run stats", lines=3)
        run.click(segment, [video, click_frame, click_xy], [out, stats])
    return demo


if __name__ == "__main__":
    build_ui().launch()
