"""Application entry point for the Gradio Web Interface."""
import os
import gradio as gr
from config.settings import settings
from src.core.logger import get_logger
from src.ui.gradio_app import create_gradio_app, HEAD_SCRIPT

logger = get_logger("app_launcher")

# Build Gradio demo application
demo = create_gradio_app()

def main():
    logger.info("Initializing Smart Document Reader...")
    try:
        settings.validate_keys()
    except Exception as e:
        logger.warning(f"Key validation warning: {e}. Ensure API keys are set in environment variables.")

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    )

    port = int(os.getenv("PORT", "7860"))
    logger.info(f"Launching Gradio UI on port {port}...")

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=theme,
        head=HEAD_SCRIPT,
        footer_links=[],
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()
