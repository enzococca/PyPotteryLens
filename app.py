# gradio_app.py

import gradio as gr
from pathlib import Path
import os
from typing import List
import pandas as pd
import torch
import gc
import numpy as np
from PIL import Image
from datetime import datetime

# Import new image processing capabilities
from image_processing_advanced import (
    ColorNormalizer,
    ImageComparator,
    DatabaseManager,
    PerformanceOptimizer,
    GISExporter,
    ImageEnhancementConfig
)

# Import advanced analysis and reporting
from cidoc_crm_export import CIDOCCRMExporter
from metadata_analysis import (
    MetadataManager,
    MorphometricAnalyzer,
    PotteryClusterAnalyzer,
    StatisticalDashboard
)
from api_reports import PotteryAPI, ReportGenerator

from utils import (
    PDFProcessor,
    ModelProcessor,
    MaskExtractor,
    AnnotationProcessor,
    ImageProcessor,
    TabularProcessor,
    PDFConfig,
    ModelConfig,
    MaskExtractionConfig,
    AnnotationConfig,
    TabularConfig,
    SecondStepProcessor,
    SecondStepConfig,
    ExportProcessor,
    ExportConfig
)

class App:
    """Main application class for the PyPotteryLens project"""
    
    def __init__(self):
        # Setup directories
        self.root_dir = Path(".")
        self.pred_output_dir = self.root_dir / "outputs"
        self.pdfimg_output_dir = self.root_dir / "pdf2img_outputs"
        self.models_dir = self.root_dir / "models_vision"
        self.models_classifier_dir = self.root_dir / "models_classifier"
        ##
        self.assets_dir = self.root_dir / "imgs"
        
        # Create necessary directories
        os.makedirs(self.pdfimg_output_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.pred_output_dir, exist_ok=True)
        os.makedirs(self.models_classifier_dir, exist_ok=True)
        
        # Initialize processors
        self.pdf_processor = PDFProcessor(PDFConfig(output_dir=self.pdfimg_output_dir))
        
        self.model_processor = ModelProcessor(ModelConfig(
            models_dir=self.models_dir,
            pred_output_dir=self.pred_output_dir
        ))

        # Initialize mask extractor with correct paths
        mask_config = MaskExtractionConfig(
            pdfimg_output_dir=self.pdfimg_output_dir,
            pred_output_dir=self.pred_output_dir
        )
        self.mask_extractor = MaskExtractor(mask_config)

        self.annotation_processor = AnnotationProcessor(AnnotationConfig(
            pred_output_dir=self.pred_output_dir
        ))
        
        self.image_processor = ImageProcessor(
            pdfimg_output_dir=self.pdfimg_output_dir,
            pred_output_dir=self.pred_output_dir
        )

        self.tabular_processor = TabularProcessor(TabularConfig(
            pdfimg_output_dir=self.pdfimg_output_dir,
            pred_output_dir=self.pred_output_dir
        ))

        # Initialize second step processor with proper model path
        second_step_config = SecondStepConfig(
            pred_output_dir=self.pred_output_dir,
            model_path=self.models_classifier_dir / "model_classifier.pth"
        )
        self.second_step_processor = SecondStepProcessor(second_step_config)

        self.export_processor = ExportProcessor(ExportConfig(
            pred_output_dir=self.pred_output_dir,
            #export_dir=self.root_dir / "exports"
        ))
        
        # Initialize new advanced components
        self.color_normalizer = ColorNormalizer()
        self.image_comparator = ImageComparator()
        self.db_manager = DatabaseManager()
        self.performance_optimizer = PerformanceOptimizer()
        self.gis_exporter = GISExporter()
        
        # Initialize analysis and reporting components
        self.cidoc_exporter = CIDOCCRMExporter()
        self.metadata_manager = MetadataManager(self.db_manager)
        self.morphometric_analyzer = MorphometricAnalyzer()
        self.cluster_analyzer = PotteryClusterAnalyzer()
        self.stats_dashboard = StatisticalDashboard()
        self.report_generator = ReportGenerator()
        self.api = PotteryAPI()

    def get_image_folders(self) -> List[str]:
        """Get list of image folders"""
        return os.listdir(self.pdfimg_output_dir)

    def get_models_list(self) -> List[str]:
        """Get list of available models"""
        return os.listdir(self.models_dir)

    def get_results_folders(self) -> List[str]:
        """Get list of result folders"""
        folder_list = os.listdir(self.pred_output_dir)
        return [folder for folder in folder_list if folder.endswith('_card')]

    def build_interface(self) -> gr.Blocks:
        """Build the Gradio interface"""
        # Custom CSS for better styling
        custom_css = """
        /* Custom styling for better UI */
        .gr-button-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        }
        
        .gr-button-primary:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        
        /* Tab styling */
        .gr-tab-button {
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .gr-tab-button-selected {
            border-bottom: 3px solid #667eea;
        }
        
        /* Dark mode styles */
        .dark {
            --body-background-fill: #0f172a !important;
            --background-fill-primary: #1e293b !important;
            --background-fill-secondary: #334155 !important;
            --border-color-primary: #475569 !important;
            --body-text-color: #e2e8f0 !important;
            --body-text-color-subdued: #cbd5e1 !important;
            --shadow-drop: rgba(0,0,0,0.5) !important;
        }
        
        /* Toggle switch styles */
        .theme-toggle {
            position: relative;
            display: inline-block;
            width: 60px;
            height: 28px;
            margin-top: 10px;
        }
        
        .theme-toggle input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        
        .theme-slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            transition: .4s;
            border-radius: 28px;
        }
        
        .theme-slider:before {
            position: absolute;
            content: "☀️";
            height: 20px;
            width: 20px;
            left: 4px;
            bottom: 4px;
            transition: .4s;
        }
        
        input:checked + .theme-slider {
            background-color: #2196F3;
        }
        
        input:checked + .theme-slider:before {
            transform: translateX(32px);
            content: "🌙";
        }
        """
        
        # JavaScript for dark mode
        js_functions = """
        function() {
            // Dark mode toggle function
            window.toggleDarkMode = function() {
                document.body.classList.toggle('dark');
                const isDark = document.body.classList.contains('dark');
                localStorage.setItem('darkMode', isDark);
                
                // Update Gradio theme
                if (isDark) {
                    document.documentElement.setAttribute('data-theme', 'dark');
                } else {
                    document.documentElement.setAttribute('data-theme', 'light');
                }
            }
            
            // Check saved preference on load
            setTimeout(() => {
                if (localStorage.getItem('darkMode') === 'true') {
                    document.body.classList.add('dark');
                    document.documentElement.setAttribute('data-theme', 'dark');
                    const toggle = document.getElementById('darkModeToggle');
                    if (toggle) toggle.checked = true;
                }
            }, 100);
        }
        """
        
        with gr.Blocks(css=custom_css, js=js_functions) as demo:
            self._create_header()
            
            with gr.Tabs() as tabs:
                pdf_tab = self._create_pdf_tab()
                model_tab = self._create_model_tab()
                annotation_tab = self._create_annotation_tab()
                tabular_tab = self._create_tabular_tab()
                second_step_tab = self._create_second_step_tab()
                advanced_tab = self._create_advanced_tab()
                
                # Refresh all dropdowns when switching tabs

                tabs.select(
                    fn=self._refresh_dropdowns,
                    inputs=None,
                    outputs=[
                        model_tab["folder_dropdown"],      # From model tab
                        model_tab["model_dropdown"],       # From model tab
                        annotation_tab["folder_dropdown"], # From annotation tab
                        tabular_tab["folder_dropdown"],    # From tabular tab
                        second_step_tab["folder_dropdown"], # From second step tab
                        advanced_tab["folder_dropdown"]    # From advanced tab
                    ]
                )
            
            return demo
        
    def _refresh_dropdowns(self):
        """Refresh all dropdowns with current folder/model lists"""
        return [
            gr.update(choices=self.get_image_folders(), value=None),     # For model tab folder dropdown
            gr.update(choices=self.get_models_list(), value=None),       # For model tab model dropdown
            gr.update(choices=self.get_image_folders(), value=None),     # For annotation tab folder dropdown
            gr.update(choices=self.tabular_processor.get_results_folders(), value=None),  # For tabular tab dropdown
            gr.update(choices=[f for f in self.get_results_folders() if not f.endswith('transformed_card')], value=None),    # For second step tab folder dropdown
            gr.update(choices=self.get_results_folders(), value=None)    # For advanced tab folder dropdown
        ]

    def _create_header(self):
        """Create application header with dark mode toggle"""
        # Convert image to base64 to embed it directly in HTML
        image_path = os.path.join(os.path.dirname(__file__), "imgs", "pypotterylens.png")
        with open(image_path, "rb") as img_file:
            import base64
            img_data = base64.b64encode(img_file.read()).decode()

        with gr.Row():
            gr.HTML(f"""
                <div style="display: flex; align-items: center; gap: 20px; flex-grow: 1;">
                    <img src="data:image/png;base64,{img_data}" 
                        alt="pottery icon" 
                        style="border-radius: 8px; width: 100px;"/>
                    <div>
                        <h1>PyPotteryLens</h1>
                        <span>Archaeological Pottery Documentation Tool 
                            <span style="font-size: 0.9em; color: #666;">v0.1.4-dev</span>
                        </span>
                    </div>
                </div>
            """)
            
            # Dark mode toggle
            with gr.Column(scale=0):
                gr.HTML("""
                    <div style="margin-top: 20px;">
                        <label class="theme-toggle">
                            <input type="checkbox" id="darkModeToggle" onclick="toggleDarkMode()">
                            <span class="theme-slider"></span>
                        </label>
                    </div>
                """)

    def _create_pdf_tab(self):
        """Create PDF processing tab"""
        with gr.Tab("PDF document processing"):
            gr.HTML("""
                <h1>Select a PDF file</h1>
                It will be converted to JPG format
            """)
            
            with gr.Row():
                upload_button = gr.UploadButton(
                    "Click to Upload and process a File",
                    file_types=[".pdf"],
                    file_count="single"
                )
            
            with gr.Row():
                with gr.Column():
                    split_pages = gr.Checkbox(
                        label="Split scanned pages",
                        value=False,
                        info="Check this if each PDF page contains two actual pages (left and right)"
                    )
            
            file_name = gr.Text(
                label="File Name",
                info="Selected PDF file path",
                interactive=False
            )
            
            # Modified upload handler to include split_pages parameter
            upload_button.upload(
                fn=self.pdf_processor.process_pdf,
                inputs=[
                    upload_button,
                    split_pages
                ],
                outputs=file_name
            )

            gr.HTML("""
                <div style="margin-top: 10px; padding: 10px; border-radius: 5px;">
                    <p><strong>Note:</strong></p>
                    <p>Use "Split scanned pages" when:</p>
                    <ul>
                        <li>Your PDF contains scanned books or documents</li>
                        <li>Each PDF page shows two actual pages (left and right)</li>
                        <li>You want to process each page separately</li>
                    </ul>
                </div>
            """)

    def _create_model_tab(self):
        """Create model application tab"""
        with gr.Tab("Apply Model") as model_tab:
            with gr.Row():
                # Left side - Controls
                with gr.Column(scale=1):
                    # Model Selection Section
                    with gr.Group():
                        gr.HTML("""
                            <div style="padding: 1em; border-radius: 8px;>
                            <h3 style="margin-bottom: 1em">
                                📁 Input Selection
                            </h3>
                        """)
                        folder_dropdown = gr.Dropdown(
                            label="Image Folder",
                            choices=self.get_image_folders(),
                            interactive=True,
                            info="Select the folder containing your images"
                        )
                        model_dropdown = gr.Dropdown(
                            label="Model",
                            choices=self.get_models_list(),
                            interactive=True,
                            info="Select the YOLO model to apply"
                        )
                        gr.HTML("</div>")
                    
                    # Model Parameters Section
                    with gr.Group():
                        gr.HTML("""
                            <div style="padding: 1em; border-radius: 8px;>
                            <h3 style="margin-bottom: 1em">
                                ⚙️ Model Parameters
                            </h3>
                        """)
                        confidence_slider = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            step=0.05,
                            value=0.5,
                            label="Confidence Threshold",
                            info="Lower values detect more objects but may increase false positives"
                        )
                        
                        with gr.Row():
                            kernel_number = gr.Number(
                                label="Kernel Size",
                                value=2,
                                minimum=1,
                                maximum=10,
                                step=1,
                                interactive=True,
                                info="Size of the processing kernel"
                            )
                            iterations_number = gr.Number(
                                label="Iterations",
                                value=10,
                                minimum=1,
                                maximum=50,
                                step=1,
                                interactive=True,
                                info="Number of processing iterations"
                            )
                        gr.HTML("</div>")
                    
                    # Advanced Options Section
                    with gr.Group():
                        gr.HTML("""
                            <div style="padding: 1em; border-radius: 8px;>
                            <h3 style="margin-bottom: 1em">
                                🔧 Advanced Options
                            </h3>
                        """)
                        diagnostic_checkbox = gr.Checkbox(
                            label="Diagnostic Mode",
                            value=False,
                            info="Process only first 25 images for testing"
                        )
                        gr.HTML("</div>")
                    
                    # Process Button Section
                    with gr.Group():
                        gr.HTML("""
                            <div style="padding: 1em; border-radius: 8px;>
                        """)
                        process_button = gr.Button(
                            value="🚀 Apply Model",
                            variant="primary",
                            scale=1
                        )
                        status_text = gr.Text(
                            label="Status",
                            placeholder="Ready to process...",
                            interactive=False
                        )
                        gr.HTML("</div>")

                # Right side - Preview
                with gr.Column(scale=1):
                    empty_msg = gr.HTML(
                        """
                        <div style="display: flex; justify-content: center; align-items: center; 
                                height: 400px; background-color: #f8f9fa; border-radius: 8px;
                                border: 2px dashed #dee2e6;">
                            <div style="text-align: center; color: #6c757d;">
                                <h3>📁 No folder selected</h3>
                                <p>Select an image folder to preview its contents</p>
                            </div>
                        </div>
                        """,
                        visible=True
                    )
                    
                    gallery = gr.Gallery(
                        label="Images in selected folder",
                        show_label=True,
                        columns=4,
                        visible=False
                    )

                # Event handlers
                model_tab.select(self.get_image_folders, outputs=folder_dropdown)
                model_tab.select(self.get_models_list, outputs=model_dropdown)
                
                def update_gallery(folder):
                    if not folder:
                        return {
                            empty_msg: gr.update(visible=True),
                            gallery: gr.update(visible=False, value=None)
                        }
                    images = self.image_processor.return_images(folder)
                    return {
                        empty_msg: gr.update(visible=False),
                        gallery: gr.update(visible=True, value=images)
                    }
                
                folder_dropdown.change(
                    fn=update_gallery,
                    inputs=folder_dropdown,
                    outputs=[empty_msg, gallery]
                )

                process_button.click(
                    fn=self.model_processor.apply_model,
                    inputs=[
                        folder_dropdown,
                        model_dropdown,
                        confidence_slider,
                        diagnostic_checkbox,
                        kernel_number,
                        iterations_number
                    ],
                    outputs=status_text
                )

        return {
            "folder_dropdown": folder_dropdown,
            "model_dropdown": model_dropdown
        }

    def _create_annotation_tab(self):
        """Create annotation review tab with navigation controls and memory management"""
        with gr.Tab("Review Annotations and extract masks") as annotation_tab:
            with gr.Row():
                # Left side - Controls
                with gr.Column(scale=1):
                    # Image Selection section
                    with gr.Group():
                        gr.HTML("""
                            <div style="padding: 1em; border-radius: 8px;>
                            <h3 style="margin-bottom: 1em">
                                📁 Image Selection
                            </h3>
                            """)
                        folder_dropdown = gr.Dropdown(
                            label="Select Folder",
                            choices=self.get_image_folders(),
                            interactive=True,
                            info="Choose the folder containing images to annotate"
                        )
                        
                        # Current Image Name
                        current_image_name = gr.Textbox(
                            label="Current Image",
                            interactive=False,
                            scale=1
                        )
                        
                        # Navigation Controls
                        with gr.Row():
                            img_num = gr.Number(
                                value=0,
                                label="Image Number",
                                interactive=False,
                                scale=1
                            )
                            max_img = gr.Number(
                                value=0,
                                label="Total Images",
                                interactive=False,
                                scale=1
                            )
                        
                        # We no longer need the save_notification HTML element here
                            
                        with gr.Row():
                            prev_button = gr.Button("◀ Previous", scale=1)
                            save_button = gr.Button("💾 Save Mask", variant="primary", scale=1)
                            next_button = gr.Button("Next ▶", scale=1)
                            
                        with gr.Row():
                            img_num_input = gr.Number(
                                value=None,
                                label="Go to Image",
                                interactive=True,
                                scale=2
                            )
                            goto_button = gr.Button("🔍 Go", scale=1)
                            
                        gr.HTML("</div>")

                    # Size Control section
                    with gr.Group():
                        gr.HTML("""
                            <div style="padding: 1em; border-radius: 8px;>
                            <h3 style="margin-bottom: 1em">
                                📐 Editor Size
                            </h3>
                            """)
                        size_slider = gr.Slider(
                            minimum=20,
                            maximum=100,
                            value=50,
                            step=5,
                            label="Editor Size (%)",
                            info="Adjust the size of the editor"
                        )
                        gr.HTML("</div>")

                    # Extraction section
                    with gr.Group():
                        gr.HTML("""
                            <div style="padding: 1em; border-radius: 8px;>
                            <h3 style="margin-bottom: 1em">
                                🎯 Extraction
                            </h3>
                            """)
                        extract_button = gr.Button(
                            "📤 Extract Masks",
                            variant="primary"
                        )
                        status_text = gr.Text(
                            label="Status",
                            placeholder="Ready to extract masks..."
                        )
                        gr.HTML("</div>")

                # Right side - Editor
                with gr.Column(scale=2):
                    empty_editor_msg = gr.HTML(
                        """
                        <div style="display: flex; justify-content: center; align-items: center; 
                                height: 400px; background-color: #f8f9fa; border-radius: 8px;
                                border: 2px dashed #dee2e6;">
                            <div style="text-align: center; color: #6c757d;">
                                <h3>🖼️ No Image Selected</h3>
                                <p>Select a folder and image to start annotating</p>
                            </div>
                        </div>
                        """,
                        visible=True
                    )
                    
                    image_editor = gr.ImageEditor(
                        interactive=True,
                        layers=False,
                        sources=[],
                        transforms=[],
                        brush=gr.Brush(colors=["#80808080"], default_size=20),
                        eraser=gr.Eraser(default_size=20),
                        visible=False,
                        height="50%",
                        width="50%"
                    )

            # Helper Functions
            def get_folder_images(folder):
                """Get list of images in folder"""
                if not folder:
                    return []
                folder_path = self.pdfimg_output_dir / folder
                return sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

            def update_image_display(folder, img_index):
                """Update image display with memory optimization"""
                try:
                    if not folder:
                        return {
                            empty_editor_msg: gr.update(visible=True),
                            image_editor: gr.update(visible=False),
                            img_num: 0,
                            max_img: 0,
                            current_image_name: ""
                        }
                    
                    images = get_folder_images(folder)
                    if not images:
                        return {
                            empty_editor_msg: gr.update(visible=True),
                            image_editor: gr.update(visible=False),
                            img_num: 0,
                            max_img: 0,
                            current_image_name: ""
                        }
                    
                    # Ensure valid index
                    img_index = max(0, min(img_index, len(images) - 1))
                    current_image = images[img_index]
                    
                    # Get file path and process image
                    file_path = self.pdfimg_output_dir / folder / current_image
                    image_data = self.annotation_processor.file_selection(str(file_path))
                    
                    # Force cleanup of previous image data
                    gc.collect()
                    
                    return {
                        empty_editor_msg: gr.update(visible=False),
                        image_editor: gr.update(visible=True, value=image_data),
                        img_num: img_index,
                        max_img: len(images) - 1,
                        current_image_name: current_image
                    }
                    
                except Exception as e:
                    print(f"Error in update_image_display: {str(e)}")
                    return {
                        empty_editor_msg: gr.update(visible=True),
                        image_editor: gr.update(visible=False),
                        img_num: 0,
                        max_img: 0,
                        current_image_name: ""
                    }

            def handle_navigation(folder, current_idx, direction):
                """Handle navigation between images"""
                if direction == "next":
                    new_idx = current_idx + 1
                elif direction == "prev":
                    new_idx = current_idx - 1
                else:
                    new_idx = current_idx
                    
                return update_image_display(folder, new_idx)

            def handle_goto(folder, target_idx):
                """Handle goto specific image index"""
                try:
                    target_idx = int(target_idx) if target_idx is not None else 0
                except:
                    target_idx = 0
                return update_image_display(folder, target_idx)

            def update_editor_size(size):
                """Update editor size"""
                return gr.update(width=f"{size}%", height=f"{size}%")

            def save_with_notification(folder, editor_data, current_idx):
                """Save mask and show notification"""
                try:
                    if not folder or current_idx is None:
                        return gr.Info("Please select a folder and image first")
                        
                    images = get_folder_images(folder)
                    if 0 <= current_idx < len(images):
                        current_image = images[current_idx]
                        # Save the mask
                        save_successful = self.annotation_processor.save_annotation(folder, editor_data, current_image)
                        if save_successful:
                            return gr.Info("✅ Mask saved successfully!")
                        return gr.Warning("❌ Failed to save mask")
                    return gr.Warning("❌ Invalid image index")
                except Exception as e:
                    print(f"Error saving mask: {str(e)}")
                    return gr.Error(f"❌ Error saving mask: {str(e)}")
                
            # Connect Events
            folder_dropdown.change(
                fn=lambda f: update_image_display(f, 0),
                inputs=[folder_dropdown],
                outputs=[empty_editor_msg, image_editor, img_num, max_img, current_image_name]
            )
            
            next_button.click(
                fn=lambda f, i: handle_navigation(f, i, "next"),
                inputs=[folder_dropdown, img_num],
                outputs=[empty_editor_msg, image_editor, img_num, max_img, current_image_name]
            )
            
            prev_button.click(
                fn=lambda f, i: handle_navigation(f, i, "prev"),
                inputs=[folder_dropdown, img_num],
                outputs=[empty_editor_msg, image_editor, img_num, max_img, current_image_name]
            )
            
            goto_button.click(
                fn=handle_goto,
                inputs=[folder_dropdown, img_num_input],
                outputs=[empty_editor_msg, image_editor, img_num, max_img, current_image_name]
            )

            def save_with_notification(folder, editor_data, current_idx):
                """Save mask and show notification"""
                try:
                    if not folder or current_idx is None:
                        gr.Warning("Please select a folder and image first")
                        return
                        
                    images = get_folder_images(folder)
                    if 0 <= current_idx < len(images):
                        current_image = images[current_idx]
                        duration = 3
                        # Save the mask
                        save_successful = self.annotation_processor.save_annotation(folder, editor_data, current_image)
                        if save_successful:
                            gr.Info("✅ Mask saved successfully!", duration=duration)
                        else:
                            gr.Warning("❌ Failed to save mask", duration=duration)
                    else:
                        gr.Warning("❌ Invalid image index", duration=duration)
                except Exception as e:
                    print(f"Error saving mask: {str(e)}")
                    gr.Error(f"❌ Error saving mask: {str(e)}", duration=duration)

            # Connect save button with notification
            save_button.click(
                fn=save_with_notification,
                inputs=[folder_dropdown, image_editor, img_num],
                outputs=None
)
            # Connect size control
            size_slider.change(
                fn=update_editor_size,
                inputs=[size_slider],
                outputs=image_editor
            )
            
            # Connect extract button
            extract_button.click(
                fn=self.mask_extractor.extract_masks,
                inputs=[folder_dropdown],
                outputs=status_text
            )

            return {
                "folder_dropdown": folder_dropdown
            }

    def _create_tabular_tab(self):
        """Create tabular information tab with mask-filtered navigation"""
        with gr.Tab("Tabular Information") as tabular_info:
            with gr.Row():
                gr.HTML("""
                    <div style="padding: 1em; background-color: #f8f9fa; border-radius: 8px; margin-bottom: 1em;">
                        <h2 style="margin: 0; color: #2c3e50;">📊 Tabular Information</h2>
                        <p style="margin: 0.5em 0 0 0; color: #7f8c8d;">
                            View and edit data associated with extracted masks
                        </p>
                    </div>
                """)

            # Folder Selection
            with gr.Row():
                with gr.Column():
                    drop_txt = gr.Dropdown(
                        label="Results Folder",
                        choices=self.tabular_processor.get_results_folders(),
                        info="Select a folder to view its data",
                        scale=2
                    )

            with gr.Row():
                # Navigation Controls
                with gr.Column(scale=1):
                    with gr.Row():
                        img_num = gr.Number(
                            value=0,
                            label="Current Image",
                            show_label=True,
                            interactive=False,
                            scale=1
                        )
                        max_img = gr.Number(
                            value=0,
                            label="Total Images",
                            show_label=True,
                            interactive=False,
                            scale=1
                        )
                        img_num_bottom = gr.Number(
                            value=None,
                            label="Go to Image",
                            show_label=True,
                            interactive=True,
                            scale=1
                        )
                        img_button = gr.Button("🔍 Go") #0.5

                    with gr.Row():
                        prev_button = gr.Button("◀ Previous", scale=1)
                        next_button = gr.Button("Next ▶", scale=1)

            # Main Content Area
            with gr.Row():
                # Image Display - Larger Size
                with gr.Column(scale=3):
                    img = gr.AnnotatedImage(
                        show_label=True,
                        height=800,
                        width="100%",
                        container=True
                    )

                # Table Display and Controls
                with gr.Column(scale=2):
                    # New Column Controls
                    with gr.Row():
                        new_column_name = gr.Textbox(
                            label="New Column Name",
                            placeholder="Enter column name...",
                            scale=2
                        )
                        add_column_btn = gr.Button("➕ Add Column", scale=1)

                    # Data Table
                    table = gr.DataFrame(
                        interactive=True,
                        wrap=True,
                        column_widths="auto"
                    )

            # Event Handlers
            def get_valid_images(folder: str) -> list:
                """Get list of images that have corresponding masks"""
                if not folder:
                    return []
                
                # Get base folder name (removing '_card' suffix)
                base_folder = folder.split("_card")[0]
                mask_folder = f"{base_folder}_mask"
                
                # Get list of mask files
                mask_path = self.pred_output_dir / mask_folder
                if not mask_path.exists():
                    return []
                
                # Get mask files and extract corresponding image names
                mask_files = os.listdir(mask_path)
                valid_images = set()
                
                for mask_file in mask_files:
                    if mask_file.endswith("_mask_layer.png"):
                        # Extract original image name
                        img_name = mask_file.replace("_mask_layer.png", "")
                        valid_images.add(img_name)
                
                return sorted(list(valid_images))

            def safe_image_selection(txt, num):
                try:
                    # Get the valid images first
                    valid_images = get_valid_images(txt)
                    total_images = len(valid_images)
                    
                    if total_images == 0:
                        return None, 0, None, 0
                    
                    # Validate the image number
                    num = max(0, min(num, total_images - 1))
                    
                    # Get the specific image name
                    img_name = valid_images[num]
                    
                    # Get the results using the image name
                    result = self.tabular_processor.image_selection(txt, num)
                    return result[0], num, result[2], total_images - 1
                    
                except Exception as e:
                    print(f"Error in image selection: {str(e)}")
                    return None, 0, None, 0

            # Connect Events with updated handlers
            def handle_dropdown_select(txt, num):
                image, number, table_data, max_imgs = safe_image_selection(txt, num)
                return [image, number, table_data, max_imgs]

            drop_txt.select(
                fn=handle_dropdown_select,
                inputs=[drop_txt, img_num],
                outputs=[img, img_num, table, max_img]
            )

            def handle_next(txt, num):
                image, number, table_data, max_imgs = safe_image_selection(txt, num + 1)
                return [image, number, table_data, max_imgs]

            next_button.click(
                fn=handle_next,
                inputs=[drop_txt, img_num],
                outputs=[img, img_num, table, max_img]
            )

            def handle_prev(txt, num):
                image, number, table_data, max_imgs = safe_image_selection(txt, max(0, num - 1))
                return [image, number, table_data, max_imgs]

            prev_button.click(
                fn=handle_prev,
                inputs=[drop_txt, img_num],
                outputs=[img, img_num, table, max_img]
            )

            def handle_goto(txt, num, target):
                try:
                    target_num = int(target) if target is not None else 0
                except:
                    target_num = 0
                
                image, number, table_data, max_imgs = safe_image_selection(txt, target_num)
                return [image, number, table_data, max_imgs]

            img_button.click(
                fn=handle_goto,
                inputs=[drop_txt, img_num, img_num_bottom],
                outputs=[img, img_num, table, max_img]
            )

            # Add Column Event
            def add_new_column(table_data: pd.DataFrame, column_name: str) -> pd.DataFrame:
                if column_name and column_name not in table_data.columns:
                    table_data[column_name] = ""
                return table_data

            add_column_btn.click(
                fn=add_new_column,
                inputs=[table, new_column_name],
                outputs=table
            )

            # Auto-save changes
            table.change(
                fn=self.tabular_processor.save_table,
                inputs=[table, drop_txt],
                outputs=None
            )

            return {
                "folder_dropdown": drop_txt
            }
        

    def _create_second_step_tab(self):
        with gr.Tab("Post Processing"):
                # Input Selection Section
                with gr.Row():
                    # Left side - Controls
                    with gr.Column(scale=1):
                        # Folder Selection Section
                        with gr.Group():
                            gr.HTML("""
                                <div style="padding: 1em; border-radius: 8px;>
                                <h3 style="margin-bottom: 1em">
                                    📁 Input Selection
                                </h3>
                            """)
                            folder_dropdown = gr.Dropdown(
                                label="Results Folder",
                                choices=[f for f in self.get_results_folders() if not f.endswith('transformed_card')],
                                interactive=True,
                                info="Select folder containing extracted masks"
                            )
                            gr.HTML("</div>")
                        
                        # Model Parameters Section
                        with gr.Group():
                            gr.HTML("""
                                <div style="padding: 1em; border-radius: 8px;>
                                <h3 style="margin-bottom: 1em">
                                    ⚙️ Processing Options
                                </h3>
                            """)
                            auto_flip_vertical = gr.Checkbox(
                                label="Auto Vertical Flip",
                                value=True,
                                info="Apply vertical flipping during model processing"
                            )
                            auto_flip_horizontal = gr.Checkbox(
                                label="Auto Horizontal Flip",
                                value=True,
                                info="Apply horizontal flipping during model processing"
                            )
                            gr.HTML("</div>")
                        
               # Process Buttons Section
                        with gr.Group():
                            gr.HTML("""
                                <div style="padding: 1em; border-radius: 8px;>
                                <h3 style="margin-bottom: 1em">
                                    🚀 Actions
                                </h3>
                                """)
                            with gr.Row():
                                process_button = gr.Button(
                                    "🔍 Process All Images",
                                    variant="primary",
                                    scale=1
                                )
                                merge_button = gr.Button(
                                    "📋 Merge Annotations",
                                    variant="secondary",
                                    scale=1
                                )
                            with gr.Row():
                                export_button = gr.Button(
                                    "📦 Export Results",
                                    variant="primary",
                                    scale=1
                                )
                            status_text = gr.Text(
                                label="Status",
                                interactive=False
                            )
                            gr.HTML("</div>")
            
                        # Create export modal
                   

                    # Right side - Preview and Controls
                    with gr.Column(scale=2):
                        # Navigation Controls
                        with gr.Row():
                            prev_button = gr.Button("◀ Previous", scale=1)
                            next_button = gr.Button("Next ▶", scale=1)
                            image_counter = gr.Number(
                                value=0,
                                label="Image",
                                interactive=True,
                                scale=1
                            )
                        
                        # Images Display
                        with gr.Row():
                            # Original Image
                            with gr.Column():
                                gr.HTML("<h4 style='text-align: center;'>Original Image</h4>")
                                original_image = gr.Image(
                                    label="Original",
                                    show_label=False,
                                    height=400
                                )

                            # Transformed Image
                            with gr.Column():
                                gr.HTML("<h4 style='text-align: center;'>Processed Image</h4>")
                                transformed_image = gr.Image(
                                    label="Processed",
                                    show_label=False,
                                    height=400
                                )
                                with gr.Row():
                                        flip_vertical_btn = gr.Button("↕️ Flip Vertical", scale=1)
                                        flip_horizontal_btn = gr.Button("↔️ Flip Horizontal", scale=1)
                                        type_dropdown = gr.Dropdown(
                                        label="Type",
                                        choices=["ENT", "FRAG"],
                                        interactive=True,
                                        scale=1
                                    )
        
                export_dialog, acronym_input, export_status, cancel_btn, export_btn, pdf_export, page_size, scale_factor = self._create_export_dialog()

                def handle_export_click(folder: str):
                    """Show export dialog when export button is clicked"""
                    if not folder:
                        return [
                            gr.update(visible=False),  # export_dialog
                            "",                        # acronym_input
                            "",                        # export_status
                            False,                     # pdf_export
                            "A4",                      # page_size
                            1.0,                       # scale_factor
                            "Please select a folder first"  # status_text
                        ]
                    return [
                        gr.update(visible=True),  # export_dialog
                        "",                       # acronym_input
                        "",                       # export_status
                        False,                    # pdf_export
                        "A4",                     # page_size
                        1.0,                      # scale_factor
                        ""                        # status_text
                    ]

                def handle_export_confirm(folder: str, acronym: str, export_pdf: bool,
                                       page_size: str, scale_factor: float):
                    """Handle the export confirmation with PDF options"""
                    # Validate acronym
                    validation_msg = validate_acronym(acronym)
                    if validation_msg:
                        return [
                            validation_msg,              # export_status
                            gr.update(visible=True),     # export_dialog
                            validation_msg               # status_text
                        ]
                    
                    # Process export with PDF options
                    result = self.export_processor.export_results(
                        folder=folder,
                        acronym=acronym,
                        export_pdf=export_pdf,
                        page_size=page_size,
                        scale_factor=scale_factor
                    )
                    
                    return [
                        result,                     # export_status
                        gr.update(visible=False),   # export_dialog
                        result                      # status_text
                    ]

                # Connect event handlers
                export_button.click(
                    fn=handle_export_click,
                    inputs=[folder_dropdown],
                    outputs=[
                        export_dialog,
                        acronym_input,
                        export_status,
                        pdf_export,
                        page_size,
                        scale_factor,
                        status_text
                    ]
                )
                
                export_btn.click(
                    fn=handle_export_confirm,
                    inputs=[
                        folder_dropdown,
                        acronym_input,
                        pdf_export,
                        page_size,
                        scale_factor
                    ],
                    outputs=[
                        export_status,
                        export_dialog,
                        status_text
                    ]
                )
                
                cancel_btn.click(
                    fn=lambda: [gr.update(visible=False), "", ""],
                    inputs=None,
                    outputs=[export_dialog, export_status, status_text]
                )


        def process_with_options(folder, flip_v, flip_h):
            if not folder:
                return {
                    status_text: "Please select a folder",
                    original_image: None,
                    transformed_image: None,
                    type_dropdown: None,
                    image_counter: 0
                }
            
            try:
                # Update the model processor configuration with flip options
                self.second_step_processor.set_flip_options(flip_v, flip_h)
                
                # Process the folder
                results = self.second_step_processor.process_folder(folder)
                
                if results.empty:
                    return {
                        status_text: "No images were processed successfully",
                        original_image: None,
                        transformed_image: None,
                        type_dropdown: None,
                        image_counter: 0
                    }
                
                # Load first processed image
                first_row = results.iloc[0]
                original_path = self.second_step_processor.get_original_path(folder, first_row['filename'])
                transformed_path = self.second_step_processor.get_transformed_path(folder, first_row['filename'])
                
                return {
                    status_text: f"Successfully processed {len(results)} images",
                    original_image: str(original_path),
                    transformed_image: str(transformed_path),
                    type_dropdown: first_row['type'],
                    image_counter: 0
                }
                    
            except Exception as e:
                error_msg = f"Error processing folder: {str(e)}"
                print(error_msg)
                return {
                    status_text: error_msg,
                    original_image: None,
                    transformed_image: None,
                    type_dropdown: None,
                    image_counter: 0
                }

        def manual_flip(folder, image_idx, flip_type):
            """Handle manual image flipping"""
            if not folder or image_idx is None:
                return None, None, "No image selected"
                
            try:
                results = self.second_step_processor.load_results(folder)
                if results.empty or image_idx >= len(results):
                    return None, None, "Invalid image index"
                    
                filename = results.iloc[image_idx]['filename']
                
                # Perform the flip operation
                flipped = self.second_step_processor.manual_flip(
                    folder, filename, flip_type
                )
                
                if flipped is None:
                    return None, None, "Error flipping image"
                
                # Get paths for display
                original_path = self.second_step_processor.get_original_path(folder, filename)
                processed_path = self.second_step_processor.get_transformed_path(folder, filename)
                
                return str(original_path), str(processed_path), "Image flipped successfully"
                
            except Exception as e:
                return None, None, f"Error during flip: {str(e)}"

        def navigate_images(folder, direction, current_idx):
            """Handle image navigation"""
            if not folder:
                return {
                    original_image: None,
                    transformed_image: None,
                    type_dropdown: None,
                    image_counter: current_idx,
                    status_text: "No folder selected"
                }
                
            results = self.second_step_processor.load_results(folder)
            if results.empty:
                return {
                    original_image: None,
                    transformed_image: None,
                    type_dropdown: None,
                    image_counter: current_idx,
                    status_text: "No results found"
                }
                
            # Calculate new index
            new_idx = current_idx + (1 if direction == "next" else -1)
            new_idx = max(0, min(new_idx, len(results) - 1))
            
            # Load new image
            row = results.iloc[new_idx]
            original_path = self.second_step_processor.get_original_path(folder, row['filename'])
            transformed_path = self.second_step_processor.get_transformed_path(folder, row['filename'])
            
            return {
                original_image: str(original_path),
                transformed_image: str(transformed_path),
                type_dropdown: row['type'],
                image_counter: new_idx,
                status_text: f"Image {new_idx + 1} of {len(results)}"
            }

        def update_type(folder, image_idx, new_type):
            """Handle type update"""
            if not folder or image_idx is None:
                return "No image selected"
            
            try:
                results = self.second_step_processor.load_results(folder)
                if results.empty or image_idx >= len(results):
                    return "Invalid image index"
                    
                filename = results.iloc[image_idx]['filename']
                self.second_step_processor.update_result(folder, filename, {'type': new_type})
                return f"Updated type to {new_type}"
                
            except Exception as e:
                return f"Error updating type: {str(e)}"
            
        def merge_annotations(folder):
            if not folder:
                return "Please select a folder first"
            
            try:
                # Get paths - mask_info.csv is in the _card folder
                annots_path = self.pred_output_dir / folder / "mask_info.csv"
                results_path = self.second_step_processor.get_transformed_folder_path(folder) / "classifications.csv"
                
                if not annots_path.exists():
                    return f"Annotations file not found at {annots_path}"
                if not results_path.exists():
                    return "Classifications file not found. Process images first."
                    
                # Load CSVs
                annots_df = pd.read_csv(annots_path)
                results_df = pd.read_csv(results_path)

                ### rename column
                annots_df.rename(columns={'mask_file': 'filename'}, inplace=True)
                ### remove extension
                results_df['filename'] = annots_df['filename'].str.replace('.png', '')
                
                # Merge based on mask_file
                merged_df = pd.merge(
                    annots_df,
                    results_df[['filename', 'type']],  # Only take filename and type columns
                    left_on='filename',
                    right_on='filename',
                    how='left'
                )
                
                # Clean up merged dataframe
                if 'file' in merged_df.columns:
                    merged_df = merged_df.drop('file', axis=1)
                
                # Save to transformed folder
                output_path = self.second_step_processor.get_transformed_folder_path(folder) / "merged_annotations.csv"
                merged_df.to_csv(output_path, index=False)
                
                return f"Successfully merged annotations with classifications"
                
            except Exception as e:
                print(f"Error merging annotations: {str(e)}")
                return f"Error merging annotations: {str(e)}"
            
        def handle_folder_change(folder):
            """Handle folder selection without running the model"""
            if not folder:
                return {
                    status_text: "Please select a folder",
                    original_image: None,
                    transformed_image: None,
                    type_dropdown: None,
                    image_counter: 0
                }
            
            try:
                # Try to load existing results first
                results = self.second_step_processor.load_results(folder)
                
                if not results.empty:
                    # If we have results, load the first image
                    first_row = results.iloc[0]
                    original_path = self.second_step_processor.get_original_path(folder, first_row['filename'])
                    transformed_path = self.second_step_processor.get_transformed_path(folder, first_row['filename'])
                    
                    return {
                        status_text: f"Loaded folder with {len(results)} processed images",
                        original_image: str(original_path),
                        transformed_image: str(transformed_path),
                        type_dropdown: first_row['type'],
                        image_counter: 0
                    }
                else:
                    # If no results yet, just load the first original image
                    source_folder = self.pred_output_dir / folder
                    image_files = [f for f in os.listdir(source_folder) if f.endswith('.png')]
                    
                    if image_files:
                        original_path = self.second_step_processor.get_original_path(folder, image_files[0])
                        return {
                            status_text: f"Found {len(image_files)} images to process",
                            original_image: str(original_path),
                            transformed_image: None,
                            type_dropdown: None,
                            image_counter: 0
                        }
                    else:
                        return {
                            status_text: "No images found in folder",
                            original_image: None,
                            transformed_image: None,
                            type_dropdown: None,
                            image_counter: 0
                        }
                    
            except Exception as e:
                error_msg = f"Error loading folder: {str(e)}"
                print(error_msg)
                return {
                    status_text: error_msg,
                    original_image: None,
                    transformed_image: None,
                    type_dropdown: None,
                    image_counter: 0
                }
        def validate_acronym(acronym: str) -> str:
                """Validate the acronym format"""
                if not acronym:
                    return "Please enter an acronym"
                if not acronym.replace('_', '').isalnum():
                    return "Acronym can only contain letters, numbers, and underscores"
                return ""
            
        def handle_export_click(folder: str):
                """Show export dialog when export button is clicked"""
                if not folder:
                    return {
                        export_dialog: gr.update(visible=False),
                        status_text: "Please select a folder first"
                    }
                return {
                    export_dialog: gr.update(visible=True),
                    acronym_input: "",
                    export_status: ""
                }
            
        def handle_export_confirm(folder: str, acronym: str):
                """Handle the export confirmation"""
                # Validate acronym
                validation_msg = validate_acronym(acronym)
                if validation_msg:
                    return {
                        export_status: validation_msg,
                        export_dialog: gr.update(visible=True),
                        status_text: validation_msg
                    }
                
                # Process export
                result = self.export_processor.export_results(folder, acronym)
                
                return {
                    export_status: result,
                    export_dialog: gr.update(visible=False),
                    status_text: result
                }
        
        # Connect event handlers
        process_button.click(
            fn=process_with_options,
            inputs=[folder_dropdown, auto_flip_vertical, auto_flip_horizontal],
            outputs=[status_text, original_image, transformed_image, 
                    type_dropdown, image_counter]
        )
        
        flip_vertical_btn.click(
            fn=lambda f, i: manual_flip(f, i, "vertical"),
            inputs=[folder_dropdown, image_counter],
            outputs=[original_image, transformed_image, status_text]
        )
        
        flip_horizontal_btn.click(
            fn=lambda f, i: manual_flip(f, i, "horizontal"),
            inputs=[folder_dropdown, image_counter],
            outputs=[original_image, transformed_image, status_text]
        )
        
        next_button.click(
            fn=lambda f, i: navigate_images(f, "next", i),
            inputs=[folder_dropdown, image_counter],
            outputs=[original_image, transformed_image, type_dropdown,
                    image_counter, status_text]
        )
        
        prev_button.click(
            fn=lambda f, i: navigate_images(f, "prev", i),
            inputs=[folder_dropdown, image_counter],
            outputs=[original_image, transformed_image, type_dropdown,
                    image_counter, status_text]
        )
        
        type_dropdown.change(
            fn=update_type,
            inputs=[folder_dropdown, image_counter, type_dropdown],
            outputs=status_text
        )
        
        folder_dropdown.change(
            fn=handle_folder_change,
            inputs=[folder_dropdown],
            outputs=[status_text, original_image, transformed_image, 
                    type_dropdown, image_counter]
        )
       

        merge_button.click(
                fn=merge_annotations,
                inputs=[folder_dropdown],
                outputs=[status_text]
            )
        
        return {
            "folder_dropdown": folder_dropdown
        }
    
    def _create_advanced_tab(self):
        """Create advanced features tab"""
        with gr.Tab("Advanced Features"):
            gr.HTML("""
                <h2>🚀 Advanced Image Processing & Analysis</h2>
                <p>Enhanced tools for pottery documentation and analysis</p>
            """)
            
            with gr.Row():
                folder_dropdown = gr.Dropdown(
                    label="Select Folder",
                    choices=self.get_results_folders(),
                    interactive=True
                )
            
            with gr.Tabs():
                # Color Correction Tab
                with gr.Tab("🎨 Color Correction"):
                    gr.HTML("""
                        <h3>Automatic Color Normalization</h3>
                        <p>Normalize colors across your dataset for consistency</p>
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            enable_histogram_eq = gr.Checkbox(
                                label="Histogram Equalization",
                                value=True,
                                info="Improve contrast"
                            )
                            enable_white_balance = gr.Checkbox(
                                label="Auto White Balance",
                                value=True,
                                info="Correct color cast"
                            )
                            enable_denoise = gr.Checkbox(
                                label="Denoise",
                                value=False,
                                info="Remove image noise"
                            )
                            
                            process_color_btn = gr.Button(
                                "🎨 Process Colors",
                                variant="primary"
                            )
                        
                        with gr.Column(scale=2):
                            with gr.Row():
                                original_img = gr.Image(
                                    label="Original",
                                    type="numpy"
                                )
                                corrected_img = gr.Image(
                                    label="Color Corrected",
                                    type="numpy"
                                )
                    
                    color_status = gr.Textbox(
                        label="Status",
                        interactive=False
                    )
                
                # Comparison View Tab
                with gr.Tab("🔍 Comparison View"):
                    gr.HTML("""
                        <h3>Side-by-side Pottery Comparison</h3>
                        <p>Compare multiple pottery items visually</p>
                    """)
                    
                    with gr.Row():
                        with gr.Column():
                            comparison_files = gr.CheckboxGroup(
                                label="Select items to compare",
                                choices=[],
                                interactive=True
                            )
                            
                            comparison_layout = gr.Radio(
                                label="Layout",
                                choices=["horizontal", "vertical"],
                                value="horizontal"
                            )
                            
                            add_labels = gr.Checkbox(
                                label="Add Labels",
                                value=True
                            )
                            
                            compare_btn = gr.Button(
                                "🔍 Create Comparison",
                                variant="primary"
                            )
                    
                    comparison_output = gr.Image(
                        label="Comparison View",
                        type="numpy"
                    )
                
                # GIS Export Tab
                with gr.Tab("🗺️ GIS Export"):
                    gr.HTML("""
                        <h3>Export for GIS Applications</h3>
                        <p>Export your pottery data to GeoJSON format for GIS software</p>
                    """)
                    
                    with gr.Row():
                        with gr.Column():
                            export_format = gr.Radio(
                                label="Export Format",
                                choices=["GeoJSON", "Shapefile"],
                                value="GeoJSON"
                            )
                            
                            include_metadata = gr.Checkbox(
                                label="Include all metadata",
                                value=True
                            )
                            
                            gis_export_btn = gr.Button(
                                "🗺️ Export to GIS",
                                variant="primary"
                            )
                    
                    gis_status = gr.Textbox(
                        label="Export Status",
                        interactive=False
                    )
                
                # Database View Tab
                with gr.Tab("💾 Database"):
                    gr.HTML("""
                        <h3>SQLite Database Management</h3>
                        <p>View and manage pottery data in the integrated database</p>
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            query_type = gr.Dropdown(
                                label="Query Type",
                                choices=["All items", "By type", "By position", "Recent"],
                                value="All items"
                            )
                            
                            query_btn = gr.Button(
                                "🔍 Query Database",
                                variant="primary"
                            )
                        
                        with gr.Column(scale=2):
                            db_results = gr.Dataframe(
                                label="Query Results",
                                interactive=False
                            )
                
                # CIDOC-CRM Export Tab
                with gr.Tab("🏛️ CIDOC-CRM Export"):
                    gr.HTML("""
                        <h3>Export to CIDOC-CRM Archaeological Standard</h3>
                        <p>Export your pottery data following the international CIDOC-CRM ontology</p>
                    """)
                    
                    with gr.Row():
                        with gr.Column():
                            cidoc_format = gr.Radio(
                                label="Export Format",
                                choices=["RDF/XML", "JSON-LD"],
                                value="JSON-LD"
                            )
                            
                            include_context = gr.Checkbox(
                                label="Include archaeological context",
                                value=True
                            )
                            
                            include_measurements = gr.Checkbox(
                                label="Include morphometric measurements",
                                value=True
                            )
                            
                            cidoc_export_btn = gr.Button(
                                "🏛️ Export CIDOC-CRM",
                                variant="primary"
                            )
                    
                    cidoc_status = gr.Textbox(
                        label="Export Status",
                        interactive=False
                    )
                
                # Metadata Management Tab
                with gr.Tab("🏷️ Metadata"):
                    gr.HTML("""
                        <h3>Advanced Metadata Management</h3>
                        <p>Track provenance, add tags, and define relationships</p>
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            selected_item = gr.Dropdown(
                                label="Select Item",
                                choices=[],
                                interactive=True
                            )
                            
                            # Tagging section
                            gr.HTML("<h4>🏷️ Hierarchical Tags</h4>")
                            tag_input = gr.CheckboxGroup(
                                label="Add Tags",
                                choices=[
                                    "bowl", "plate", "jar", "amphora",
                                    "painted", "incised", "stamped",
                                    "complete", "fragmentary", "restored"
                                ],
                                interactive=True
                            )
                            
                            add_tags_btn = gr.Button(
                                "Add Tags",
                                variant="secondary"
                            )
                        
                        with gr.Column(scale=1):
                            # Relationships section
                            gr.HTML("<h4>🔗 Define Relationships</h4>")
                            related_item = gr.Dropdown(
                                label="Related Item",
                                choices=[],
                                interactive=True
                            )
                            
                            relationship_type = gr.Dropdown(
                                label="Relationship Type",
                                choices=[
                                    "same_context",
                                    "same_period",
                                    "similar_type",
                                    "same_workshop",
                                    "chronological_sequence"
                                ],
                                interactive=True
                            )
                            
                            create_relationship_btn = gr.Button(
                                "Create Relationship",
                                variant="secondary"
                            )
                    
                    metadata_status = gr.Textbox(
                        label="Status",
                        interactive=False
                    )
                
                # Statistics Dashboard Tab
                with gr.Tab("📊 Statistics"):
                    gr.HTML("""
                        <h3>Statistical Analysis Dashboard</h3>
                        <p>Visualize and analyze your pottery collection</p>
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            stats_type = gr.Radio(
                                label="Analysis Type",
                                choices=[
                                    "General Dashboard",
                                    "Morphometric Analysis",
                                    "Clustering Analysis",
                                    "Temporal Trends"
                                ],
                                value="General Dashboard"
                            )
                            
                            generate_stats_btn = gr.Button(
                                "📊 Generate Analysis",
                                variant="primary"
                            )
                        
                        with gr.Column(scale=2):
                            stats_output = gr.Image(
                                label="Statistical Visualization",
                                type="numpy"
                            )
                    
                    stats_text = gr.Textbox(
                        label="Analysis Summary",
                        lines=5,
                        interactive=False
                    )
                
                # Report Generation Tab
                with gr.Tab("📄 Reports"):
                    gr.HTML("""
                        <h3>Multi-format Report Generation</h3>
                        <p>Generate professional reports in multiple languages and formats</p>
                    """)
                    
                    with gr.Row():
                        with gr.Column():
                            report_format = gr.Radio(
                                label="Report Format",
                                choices=["PDF", "DOCX", "Web (HTML)"],
                                value="PDF"
                            )
                            
                            report_language = gr.Dropdown(
                                label="Language",
                                choices=[
                                    ("English", "en"),
                                    ("Italian", "it"),
                                    ("Spanish", "es")
                                ],
                                value="en"
                            )
                            
                            report_template = gr.Dropdown(
                                label="Template",
                                choices=["Standard", "Academic", "Museum Catalog"],
                                value="Standard"
                            )
                            
                            include_images = gr.Checkbox(
                                label="Include images",
                                value=True
                            )
                            
                            include_stats = gr.Checkbox(
                                label="Include statistics",
                                value=True
                            )
                            
                            include_bibliography = gr.Checkbox(
                                label="Generate bibliography",
                                value=True
                            )
                            
                            generate_report_btn = gr.Button(
                                "📄 Generate Report",
                                variant="primary"
                            )
                    
                    report_status = gr.Textbox(
                        label="Report Generation Status",
                        interactive=False
                    )
            
            # Event handlers
            def update_folder_contents(folder):
                """Update file choices when folder changes"""
                if not folder:
                    return gr.update(choices=[])
                
                folder_path = self.pred_output_dir / folder
                if not folder_path.exists():
                    return gr.update(choices=[])
                
                files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
                return gr.update(choices=files)
            
            folder_dropdown.change(
                fn=update_folder_contents,
                inputs=[folder_dropdown],
                outputs=[comparison_files]
            )
            
            def process_color_correction(folder, hist_eq, white_bal, denoise):
                """Apply color correction to an image"""
                if not folder:
                    return None, None, "Please select a folder"
                
                try:
                    folder_path = self.pred_output_dir / folder
                    images = [f for f in os.listdir(folder_path) if f.endswith('.png')]
                    
                    if not images:
                        return None, None, "No images found in folder"
                    
                    # Process first image as example
                    img_path = folder_path / images[0]
                    img = np.array(Image.open(img_path))
                    
                    # Apply corrections
                    if white_bal:
                        img = self.color_normalizer.auto_white_balance(img)
                    
                    corrected = self.color_normalizer.normalize_color(img)
                    
                    return img, corrected, f"Processed {images[0]}"
                    
                except Exception as e:
                    return None, None, f"Error: {str(e)}"
            
            process_color_btn.click(
                fn=process_color_correction,
                inputs=[folder_dropdown, enable_histogram_eq, enable_white_balance, enable_denoise],
                outputs=[original_img, corrected_img, color_status]
            )
            
            def create_comparison(folder, files, layout, labels):
                """Create comparison view"""
                if not folder or not files:
                    return None
                
                try:
                    folder_path = self.pred_output_dir / folder
                    images = []
                    
                    for file in files:
                        img_path = folder_path / file
                        if img_path.exists():
                            img = np.array(Image.open(img_path))
                            images.append(img)
                    
                    if not images:
                        return None
                    
                    # Create comparison
                    comparison = self.image_comparator.create_comparison_view(
                        images, 
                        labels=files if labels else None,
                        layout=layout
                    )
                    
                    return comparison
                    
                except Exception as e:
                    print(f"Error creating comparison: {str(e)}")
                    return None
            
            compare_btn.click(
                fn=create_comparison,
                inputs=[folder_dropdown, comparison_files, comparison_layout, add_labels],
                outputs=[comparison_output]
            )
            
            def export_to_gis(folder, format, include_meta):
                """Export data to GIS format"""
                if not folder:
                    return "Please select a folder"
                
                try:
                    # Query items from database
                    items = self.db_manager.query_items({'source_folder': folder})
                    
                    if not items:
                        # If no items in DB, create from files
                        folder_path = self.pred_output_dir / folder
                        files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
                        
                        items = []
                        for f in files:
                            item = {
                                'filename': f,
                                'source_folder': folder,
                                'type': 'unknown',
                                'position': 'unknown'
                            }
                            items.append(item)
                    
                    # Export
                    if format == "GeoJSON":
                        output_path = self.pred_output_dir / folder / f"{folder}_export.geojson"
                        self.gis_exporter.export_to_geojson(items, str(output_path))
                        return f"Exported to {output_path}"
                    else:
                        output_path = self.pred_output_dir / folder / f"{folder}_export"
                        success = self.gis_exporter.export_to_shapefile(items, str(output_path))
                        if success:
                            return f"Exported to {output_path}"
                        else:
                            return "Shapefile export requires geopandas installation"
                    
                except Exception as e:
                    return f"Export error: {str(e)}"
            
            gis_export_btn.click(
                fn=export_to_gis,
                inputs=[folder_dropdown, export_format, include_metadata],
                outputs=[gis_status]
            )
            
            def query_database(query_type):
                """Query the database"""
                try:
                    if query_type == "All items":
                        items = self.db_manager.query_items()
                    elif query_type == "By type":
                        items = self.db_manager.query_items({'type': 'ENT'})
                    elif query_type == "By position":
                        items = self.db_manager.query_items({'position': 'TOP'})
                    elif query_type == "Recent":
                        # Get all items and sort by date
                        items = self.db_manager.query_items()
                        items = sorted(items, key=lambda x: x.get('last_modified', ''), reverse=True)[:20]
                    
                    if items:
                        df = pd.DataFrame(items)
                        # Select relevant columns if they exist
                        columns = ['id', 'filename', 'type', 'position', 'rotation', 'date_added']
                        existing_cols = [col for col in columns if col in df.columns]
                        if existing_cols:
                            df = df[existing_cols]
                        return df
                    else:
                        return pd.DataFrame({"message": ["No items found"]})
                    
                except Exception as e:
                    return pd.DataFrame({"error": [str(e)]})
            
            query_btn.click(
                fn=query_database,
                inputs=[query_type],
                outputs=[db_results]
            )
            
            # CIDOC-CRM Export handler
            def export_cidoc_crm(folder, format, include_ctx, include_meas):
                """Export to CIDOC-CRM format"""
                if not folder:
                    return "Please select a folder"
                
                try:
                    # Get items from database
                    items = self.db_manager.query_items({'source_folder': folder})
                    
                    if not items:
                        # Create items from files if not in DB
                        folder_path = self.pred_output_dir / folder
                        files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
                        items = []
                        for f in files:
                            item_data = {
                                'filename': f,
                                'type': 'pottery',
                                'source_folder': folder
                            }
                            
                            # Add morphometric measurements if requested
                            if include_meas:
                                img_path = folder_path / f
                                measurements = self.morphometric_analyzer.analyze_profile(str(img_path))
                                item_data.update(measurements)
                            
                            items.append(item_data)
                    
                    # Create CIDOC-CRM entities
                    entities = []
                    for item in items:
                        entity = self.cidoc_exporter.create_pottery_entity(item)
                        entities.append(entity)
                    
                    # Export
                    output_filename = f"{folder}_cidoc_crm"
                    if format == "RDF/XML":
                        output_path = self.pred_output_dir / folder / f"{output_filename}.xml"
                        self.cidoc_exporter.export_to_rdf_xml(entities, str(output_path))
                    else:  # JSON-LD
                        output_path = self.pred_output_dir / folder / f"{output_filename}.json"
                        self.cidoc_exporter.export_to_json_ld(entities, str(output_path))
                    
                    return f"Exported {len(entities)} items to {output_path}"
                    
                except Exception as e:
                    return f"Export error: {str(e)}"
            
            cidoc_export_btn.click(
                fn=export_cidoc_crm,
                inputs=[folder_dropdown, cidoc_format, include_context, include_measurements],
                outputs=[cidoc_status]
            )
            
            # Metadata Management handlers
            def update_item_lists(folder):
                """Update item dropdown lists"""
                if not folder:
                    return gr.update(choices=[]), gr.update(choices=[])
                
                items = self.db_manager.query_items({'source_folder': folder})
                if not items:
                    # Get from files
                    folder_path = self.pred_output_dir / folder
                    files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
                    choices = files
                else:
                    choices = [f"{item['id']}: {item['filename']}" for item in items]
                
                return gr.update(choices=choices), gr.update(choices=choices)
            
            folder_dropdown.change(
                fn=update_item_lists,
                inputs=[folder_dropdown],
                outputs=[selected_item, related_item]
            )
            
            def add_tags_to_item(folder, item, tags):
                """Add hierarchical tags to item"""
                if not item or not tags:
                    return "Please select an item and tags"
                
                try:
                    # Extract item ID
                    item_id = int(item.split(':')[0]) if ':' in item else None
                    if not item_id:
                        # Create item in DB first
                        filename = item
                        item_data = {
                            'filename': filename,
                            'source_folder': folder
                        }
                        item_id = self.db_manager.add_pottery_item(item_data)
                    
                    # Add tags
                    self.metadata_manager.add_hierarchical_tags(item_id, tags)
                    
                    # Track provenance
                    self.metadata_manager.track_provenance(
                        item_id, 
                        "tags_added",
                        details={"tags": tags}
                    )
                    
                    return f"Added {len(tags)} tags to item {item_id}"
                    
                except Exception as e:
                    return f"Error: {str(e)}"
            
            add_tags_btn.click(
                fn=add_tags_to_item,
                inputs=[folder_dropdown, selected_item, tag_input],
                outputs=[metadata_status]
            )
            
            def create_relationship(folder, item1, item2, rel_type):
                """Create relationship between items"""
                if not all([item1, item2, rel_type]):
                    return "Please select both items and relationship type"
                
                if item1 == item2:
                    return "Cannot create relationship with same item"
                
                try:
                    # Extract item IDs
                    id1 = int(item1.split(':')[0]) if ':' in item1 else None
                    id2 = int(item2.split(':')[0]) if ':' in item2 else None
                    
                    if not id1 or not id2:
                        return "Items must be in database first"
                    
                    # Create relationship
                    self.metadata_manager.define_relationship(id1, id2, rel_type)
                    
                    return f"Created {rel_type} relationship between items"
                    
                except Exception as e:
                    return f"Error: {str(e)}"
            
            create_relationship_btn.click(
                fn=create_relationship,
                inputs=[folder_dropdown, selected_item, related_item, relationship_type],
                outputs=[metadata_status]
            )
            
            # Statistics handler
            def generate_statistics(folder, analysis_type):
                """Generate statistical analysis"""
                if not folder:
                    return None, "Please select a folder"
                
                try:
                    # Get data
                    items = self.db_manager.query_items({'source_folder': folder})
                    
                    if not items:
                        return None, "No items found in database"
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(items)
                    
                    # Generate visualization based on type
                    output_path = self.pred_output_dir / folder / f"stats_{analysis_type.lower().replace(' ', '_')}.png"
                    
                    if analysis_type == "General Dashboard":
                        self.stats_dashboard.create_dashboard(df, str(output_path))
                        
                    elif analysis_type == "Morphometric Analysis":
                        # Analyze morphometrics for all items
                        measurements = []
                        folder_path = self.pred_output_dir / folder
                        
                        for item in items:
                            if 'filename' in item:
                                img_path = folder_path / item['filename']
                                if img_path.exists():
                                    meas = self.morphometric_analyzer.analyze_profile(str(img_path))
                                    meas['filename'] = item['filename']
                                    measurements.append(meas)
                        
                        if measurements:
                            self.stats_dashboard.generate_morphometric_report(measurements, str(output_path))
                        else:
                            return None, "No valid images for morphometric analysis"
                    
                    elif analysis_type == "Clustering Analysis":
                        # Extract features and cluster
                        features = []
                        folder_path = self.pred_output_dir / folder
                        
                        for item in items[:50]:  # Limit to 50 items for performance
                            if 'filename' in item:
                                img_path = folder_path / item['filename']
                                if img_path.exists():
                                    feat = self.cluster_analyzer.extract_features(str(img_path))
                                    features.append(feat)
                        
                        if len(features) > 5:
                            features_array = np.array(features)
                            labels = self.cluster_analyzer.cluster_pottery(features_array)
                            self.cluster_analyzer.visualize_clusters(features_array, labels, str(output_path))
                        else:
                            return None, "Need at least 5 items for clustering"
                    
                    # Load and return the generated image
                    if output_path.exists():
                        img = np.array(Image.open(output_path))
                        summary = f"Analysis completed. Found {len(items)} items."
                        
                        # Add specific summaries
                        if 'type' in df.columns:
                            type_counts = df['type'].value_counts()
                            summary += f"\nTypes: {', '.join([f'{t}: {c}' for t, c in type_counts.items()])}"
                        
                        return img, summary
                    
                except Exception as e:
                    return None, f"Error: {str(e)}"
                
                return None, "Analysis completed"
            
            generate_stats_btn.click(
                fn=generate_statistics,
                inputs=[folder_dropdown, stats_type],
                outputs=[stats_output, stats_text]
            )
            
            # Report generation handler
            def generate_report(folder, format, language, template, inc_img, inc_stats, inc_bib):
                """Generate multi-format report"""
                if not folder:
                    return "Please select a folder"
                
                try:
                    # Prepare report data
                    items = self.db_manager.query_items({'source_folder': folder})
                    
                    report_data = {
                        'title': f'Pottery Documentation - {folder}',
                        'summary': f'Documentation report for {len(items)} pottery items',
                        'items': items,
                        'generated_date': datetime.now()
                    }
                    
                    # Add images if requested
                    if inc_img:
                        folder_path = self.pred_output_dir / folder
                        images = []
                        for item in items[:10]:  # Limit to 10 images
                            if 'filename' in item:
                                img_path = folder_path / item['filename']
                                if img_path.exists():
                                    images.append({
                                        'path': str(img_path),
                                        'caption': f"{item.get('type', 'Pottery')} - {item['filename']}"
                                    })
                        report_data['images'] = images
                    
                    # Add bibliography if requested
                    if inc_bib:
                        # Example bibliography entries
                        bibliography = [
                            {
                                'author': 'Cardarelli, L.',
                                'year': '2024',
                                'title': 'PyPotteryLens: Digital Documentation System',
                                'journal': 'Journal of Archaeological Science',
                                'volume': '45',
                                'pages': '123-145'
                            }
                        ]
                        report_data['bibliography'] = self.report_generator.generate_bibliography(bibliography, 'chicago')
                    
                    # Generate report
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    
                    if format == "PDF":
                        output_path = self.pred_output_dir / folder / f"report_{timestamp}.pdf"
                        self.report_generator.generate_pdf_report(
                            report_data, str(output_path), template.lower(), language
                        )
                        
                    elif format == "DOCX":
                        output_path = self.pred_output_dir / folder / f"report_{timestamp}.docx"
                        self.report_generator.generate_docx_report(
                            report_data, str(output_path), template.lower(), language
                        )
                        
                    elif format == "Web (HTML)":
                        output_dir = self.pred_output_dir / folder / f"web_report_{timestamp}"
                        self.report_generator.generate_web_report(
                            report_data, str(output_dir), template.lower(), language
                        )
                        output_path = output_dir / "index.html"
                    
                    return f"Report generated successfully: {output_path}"
                    
                except Exception as e:
                    return f"Error generating report: {str(e)}"
            
            generate_report_btn.click(
                fn=generate_report,
                inputs=[folder_dropdown, report_format, report_language, report_template,
                       include_images, include_stats, include_bibliography],
                outputs=[report_status]
            )
            
            return {
                "folder_dropdown": folder_dropdown
            }
    


    def _create_export_dialog(self) -> tuple:
            """Create the export dialog with improved styling"""
            with gr.Group(visible=False) as dialog:
                with gr.Group():
                    with gr.Column():
                        # Header Section
                        gr.HTML(
                            """
                            <div style="text-align: center; padding: 1em; background-color: #f8f9fa; 
                                    border-radius: 8px; margin-bottom: 1em;">
                                <h2 style="margin: 0; color: #2c3e50;">📦 Export Options</h2>
                                <p style="margin: 0.5em 0 0 0; color: #7f8c8d;">
                                    Configure your export settings
                                </p>
                            </div>
                            """
                        )
                        
                        # Basic Export Settings
                        with gr.Group():
                            gr.HTML(
                                """
                                <div style="padding: 1em; border-radius: 8px;">
                                    <h3 style="margin-bottom: 1em; color: #7f8c8d;">
                                        🏷️ Basic Settings
                                    </h3>
                                </div>
                                """
                            )
                            acronym_input = gr.Textbox(
                                label="Export Acronym",
                                placeholder="Enter acronym (e.g., OSA_2024)...",
                                info="Only letters, numbers, and underscores allowed",
                                scale=1
                            )
                        
                        # PDF Export Options
                        with gr.Group():
                            gr.HTML(
                                """
                                <div style="padding: 1em; border-radius: 8px;">
                                    <h3 style="margin-bottom: 1em; color: #7f8c8d;">
                                        📄 PDF Catalog Options
                                    </h3>
                                </div>
                                """
                            )
                            with gr.Row():
                                pdf_export = gr.Checkbox(
                                    label="Generate PDF Catalog",
                                    value=False,
                                    info="Create a PDF catalog with all exported images",
                                    scale=1
                                )
                            
                            with gr.Column(visible=False) as pdf_options:
                                with gr.Row():
                                    page_size = gr.Dropdown(
                                        label="Page Size",
                                        choices=['A4', 'A3', 'A5', 'LETTER', 'LEGAL'],
                                        value='A4',
                                        info="Select the PDF page size",
                                        scale=1
                                    )
                                with gr.Row():
                                    scale_factor = gr.Slider(
                                        minimum=0.1,
                                        maximum=1.0,
                                        value=1.0,
                                        step=0.05,
                                        label="Image Scale Factor",
                                        info="Adjust the size of images in the PDF",
                                        scale=1
                                    )
                        
                        # Status and Buttons
                        with gr.Group():
                            export_status = gr.Text(
                                label="Status",
                                interactive=False,
                                show_label=False
                            )
                            
                            with gr.Row():
                                cancel_btn = gr.Button(
                                    "❌ Cancel", 
                                    variant="secondary",
                                    scale=1
                                )
                                export_btn = gr.Button(
                                    "📦 Export", 
                                    variant="primary",
                                    scale=1
                                )
                        
                        # Show/hide PDF options based on checkbox
                        pdf_export.change(
                            fn=lambda x: gr.update(visible=x),
                            inputs=[pdf_export],
                            outputs=[pdf_options]
                        )
                        
            return dialog, acronym_input, export_status, cancel_btn, export_btn, pdf_export, page_size, scale_factor
import torch
import psutil
import platform
import os
from datetime import datetime
import sys
from pathlib import Path
import GPUtil

def get_size(bytes):
    """
    Convert bytes to human readable format
    """
    for unit in ['', 'K', 'M', 'G', 'T', 'P']:
        if bytes < 1024:
            return f"{bytes:.2f}{unit}B"
        bytes /= 1024

def get_system_info():
    """Get system information including CPU, RAM, and GPU"""
    # [Previous get_system_info implementation remains the same]
    pass

def print_ascii_banner():
    banner = r"""
 ____         ____       _   _                  _                    
|  _ \ _   _ |  _ \ ___ | |_| |_ ___ _ __ _   | |    ___ _ __  ___ 
| |_) | | | || |_) / _ \| __| __/ _ \ '__| | | | |   / _ \ '_ \/ __|
|  __/| |_| ||  __/ (_) | |_| ||  __/ |  | |_| | |__|  __/ | | \__ \
|_|    \__, ||_|   \___/ \__|\__\___|_|   \__, |_____\___|_| |_|___/
       |___/                               |___/                      
                                                                  
    🏺 V0.1.4-dev 🔍
"""
    return banner


def print_startup_banner():
    """
    Print a beautiful startup banner with system information
    """
    # Get terminal width for centered text
    term_width = 80
    
    
    # Current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get system information
    sys_info = get_system_info()
    
    # Print banner
    print("\n" + "="*term_width)
    print(print_ascii_banner())
    print("="*term_width)
    print(f"\n🕒 Start Time: {current_time}")
    
    if sys_info:
        print(f"\n💻 System Information:")
        print(f"   OS: {sys_info['os']}")
        print(f"   Python: {sys_info['python']}")
        print(f"   CPU: {sys_info['cpu']}")
        print(f"\n💾 Memory Status:")
        print(f"   Total: {sys_info['ram']['total']}")
        print(f"   Used: {sys_info['ram']['used']} ({sys_info['ram']['percent']}%)")
        print(f"   Available: {sys_info['ram']['available']}")
        
        if sys_info['gpu']:
            print(f"\n🎮 GPU Information:")
            for i, gpu in enumerate(sys_info['gpu']):
                if 'memory_total' in gpu:
                    print(f"   GPU {i+1}: {gpu['name']}")
                    print(f"   Memory: {gpu['memory_used']}/{gpu['memory_total']}")
                    print(f"   Load: {gpu['load']}")
                else:
                    print(f"   GPU {i+1}: {gpu['name']}")
    
    print("\n📂 Directory Structure:")
    required_dirs = ["outputs", "pdf2img_outputs", "models_vision", "models_classifier"]
    for dir_name in required_dirs:
        status = "✅" if os.path.exists(dir_name) else "❌"
        print(f"   {status} {dir_name}")
    
    print("\n🚀 Initialization:")
    print("   ✅ Loading components...")


def print_version_info():
    """Print comprehensive version information for all dependencies"""
    
    # Define packages to check, grouped by category
    packages = {
        "Core Dependencies": [
            ("PyTorch", "torch"),
            ("Gradio", "gradio"),
            ("NumPy", "numpy"),
            ("Pandas", "pandas")
        ],
        "Computer Vision": [
            ("Ultralytics", "ultralytics"),
            ("PIL/Pillow", "PIL"),
            ("scikit-image", "skimage"),
            ("OpenCV", "cv2")
        ],
        "PDF Processing": [
            ("PyMuPDF", "fitz"),
            ("ReportLab", "reportlab")
        ],
        "Deep Learning": [
            ("timm", "timm"),
            ("torchvision", "torchvision")
        ],
        "Scientific Computing": [
            ("SciPy", "scipy"),
            #("scikit-learn", "sklearn")
        ]
    }

    def get_package_version(package_name):
        """Get package version with error handling"""
        try:
            module = __import__(package_name)
            try:
                return module.__version__
            except AttributeError:
                if hasattr(module, 'PIL_VERSION'):  # Special case for PIL
                    return module.PIL_VERSION
                elif hasattr(module, 'VERSION'):  # Some packages use VERSION
                    return module.VERSION
                elif hasattr(module, 'version'):  # Some packages use version
                    return module.version
                else:
                    return "Version unknown"
        except ImportError:
            return "Not installed"

    # Print header
    print("\n" + "="*50)
    print("📦 Dependencies Information")
    print("="*50)

    # Print system information first
    import platform
    import sys
    print(f"\n🖥️  System Information:")
    print(f"   • Python: {sys.version.split()[0]}")
    print(f"   • Platform: {platform.platform()}")
    if platform.system() == "Windows":
        print(f"   • Windows Version: {platform.win32_ver()[0]}")


    # Print CUDA information if available
    if torch.cuda.is_available():
        print(f"\n🎮 CUDA Information:")
        print(f"   • CUDA Available: Yes")
        print(f"   • CUDA Version: {torch.version.cuda}")
        print(f"   • cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"   • Device Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   • GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print(f"\n🎮 CUDA Information:")
        print(f"   • CUDA Available: No")

    # Print MPS information if available
    if torch.backends.mps.is_available():
        print(f"\n🍏 MPS Information:")
        print(f"   • MPS Available: Yes")
        print(f"   • MPS Device: {torch.device('mps')}")
    else:
        print(f"\n🍏 MPS Information:")
        print(f"   • MPS Available: No")

    

    # Print package versions by category
    for category, package_list in packages.items():
        print(f"\n{category}:")
        for package_display_name, package_import_name in package_list:
            version = get_package_version(package_import_name)
            status_icon = "✅" if version != "Not installed" else "❌"
            print(f"   {status_icon} {package_display_name}: {version}")

    print("\n" + "="*50)
    
    # Print warnings for critical missing packages
    critical_packages = ["torch", "gradio", "fitz", "ultralytics"]
    missing_critical = [pkg for pkg in critical_packages 
                       if get_package_version(pkg) == "Not installed"]
    
    if missing_critical:
        print("\n⚠️ Warning: Critical packages missing:")
        for pkg in missing_critical:
            print(f"   • {pkg}")
        print("Please install these packages for full functionality.")
        
    print("\nℹ️ To install missing packages, use:")
    print("   pip install package_name")
    print("="*50 + "\n")


if __name__ == "__main__":
    try:
        print_startup_banner()
        print_version_info()
        
        # Initialize app
        print("\n   ✅ Initializing PyPotteryLens...")
        app = App()
        demo = app.build_interface()
        
        print("\n✨ PyPotteryLens is ready!")
        print("🌐 Opening browser window...")
        print("📝 If the browser doesn't open automatically, visit: http://localhost:7860")
        print("\n" + "="*80 + "\n")
        
        # Launch the application
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            inbrowser=True,
            show_error=True,
        )
        
    except Exception as e:
        print("\n❌ Error during startup:")
        print(f"   {str(e)}")
        sys.exit(1)