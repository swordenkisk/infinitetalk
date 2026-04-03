"""Gradio Web UI"""
import gradio as gr
import numpy as np
from PIL import Image
import tempfile
import os

def create_gradio_app(pipeline_fn=None):
    """Create Gradio interface"""
    
    def generate_video(image, audio, num_frames, resolution):
        """Generate video from image and audio"""
        if pipeline_fn is None:
            return None, "Pipeline not loaded"
            
        try:
            # Convert inputs
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
                
            # Generate
            video_path = pipeline_fn(
                image=image,
                audio=audio,
                num_frames=num_frames,
                resolution=resolution
            )
            
            return video_path, "Success"
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    with gr.Blocks(title="InfiniteTalk - Video Dubbing") as demo:
        gr.Markdown("# InfiniteTalk")
        gr.Markdown("Sparse-Frame Video Dubbing with 4-Step Inference")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    label="Reference Image",
                    type="numpy"
                )
                audio_input = gr.Audio(
                    label="Input Audio",
                    type="filepath"
                )
                num_frames = gr.Slider(
                    minimum=30,
                    maximum=300,
                    value=120,
                    step=30,
                    label="Number of Frames"
                )
                resolution = gr.Dropdown(
                    choices=["480p", "720p"],
                    value="720p",
                    label="Resolution"
                )
                generate_btn = gr.Button("Generate Video", variant="primary")
                
            with gr.Column():
                video_output = gr.Video(label="Generated Video")
                status = gr.Textbox(label="Status")
                
        generate_btn.click(
            fn=generate_video,
            inputs=[image_input, audio_input, num_frames, resolution],
            outputs=[video_output, status]
        )
        
    return demo

if __name__ == "__main__":
    demo = create_gradio_app()
    demo.launch()
