
# `PyPotteryLens`

<div align="center">
<img src="imgs/pypotterylens.png" width="150"/>

[![Version](https://img.shields.io/badge/version-0.1.3-blue.svg)](https://lrncrd.github.io/PyPotteryInk/)
[![HuggingFace](https://img.shields.io/badge/🤗%20Models-PyPotteryLens-yellow.svg)](https://huggingface.co/lrncrd/PyPotteryLens)
[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![arXiv Preprint](https://img.shields.io/badge/arXiv-2412.11574-b31b1b.svg)](https://arxiv.org/abs/2412.11574)


</div>

As part of the **PyPottery** toolkit, `PyPotteryLens` is a Gradio desktop application for archaeological pottery drawings recording. It provides a comprehensive suite of tools for processing, detecting, analyzing, and documenting pottery fragments from scanned documents with advanced features for professional archaeological research.

## Core Features

- **PDF Processing**: Convert multi-page PDF documents to high-quality images
- **Fragment Detection**: Computer vision model for automatic pottery drawings detection
- **Manual Annotation**: Interactive interface for reviewing and adjusting detected drawings
- **Tabular Data Management**: Add tabular data to the detected drawings
- **Post Processing**: Automatic classification of pottery drawings for a standardized recording
- **User-Friendly Interface**: Intuitive GUI built with Gradio

## Advanced Features (v0.1.4)

### 🎨 Image Processing & Enhancement
- **Color Normalization**: Automatic color correction for dataset uniformity
- **Auto White Balance**: Correct color casts in scanned documents
- **Dark Mode**: Eye-friendly dark theme for extended use
- **Comparison View**: Side-by-side pottery comparisons
- **Overlay Tools**: Transparent overlays for detailed analysis

### 🏛️ Archaeological Standards & Export
- **CIDOC-CRM Export**: Export data following international archaeological standards
  - RDF/XML format
  - JSON-LD format
  - Full ontology compliance
- **GIS Integration**: 
  - GeoJSON export for QGIS/ArcGIS
  - Spatial data support
  - Coordinate system compatibility

### 💾 Database & Metadata Management
- **SQLite Database**: Robust relational database replacing CSV files
- **EXIF Preservation**: Maintain original image metadata
- **Provenance Tracking**: Complete history of modifications
- **Hierarchical Tagging**: Organized tag system for pottery classification
- **Relationship Mapping**: Define connections between pottery items

### 📊 Statistical Analysis & Visualization
- **Interactive Dashboard**: Real-time statistical visualizations
- **Morphometric Analysis**: 
  - Automatic measurements (height, diameter, volume)
  - Shape indices and ratios
  - Curvature analysis
- **Clustering Analysis**: Group similar pottery automatically
- **Trend Analysis**: Identify typological evolution over time

### 📄 Professional Reporting
- **Multi-format Export**: PDF, DOCX, and Web (HTML)
- **Multi-language Support**: English, Italian, Spanish
- **Customizable Templates**: Standard, Academic, Museum Catalog
- **Automatic Bibliography**: Chicago and APA citation styles
- **Web Publishing**: Generate static websites for online documentation

### 🌐 API & Integration
- **REST API**: Full CRUD operations for external system integration
- **Batch Processing**: Process entire collections in background
- **Asynchronous Operations**: Non-blocking batch tasks
- **CORS Support**: Cross-origin resource sharing enabled

### ⚡ Performance Optimization
- **Smart Caching**: Intelligent result caching system
- **Parallel Processing**: Multi-core CPU utilization
- **GPU Acceleration**: Support for CUDA and MPS (Apple Silicon)
- **Memory Management**: Optimized for large datasets

## Installation

### Quick Installation

1. Download the releases ZIP and extract it in a folder of your choice.

2. Download Python 3.11 from [Microsoft Store](https://www.microsoft.com/store/productId/9NRWMJP3717K?ocid=pdpshare)

3. Double click on `PyPotteryLens_WIN.bat` file. This will open an installation process. A virtual environment (`venv`) will be created and all the dependencies will be installed automatically. If CUDA is available, it will be installed as well. During the installation, basic models will be downloaded from HuggingFace. Several progress bars keep you informed about the installation process.

4. After the installation is complete, the program will be executed and the default browser will open the GUI.

> ⚠️ **Important**: If you are encountering issues with the installation, use the cleanup script:
> - **Windows**: Remove the `venv` folder and run `PyPotteryLens_WIN.bat` again
> - **Unix/macOS**: Run `sh cleanup.sh` then `sh PyPotteryLens_UNIX.sh` 


#### UNIX (Linux, MacOS)

1. Download the releases ZIP and extract it in a folder of your choice.

2. Open a terminal and move into the downloaded folder.

3. Run the following command:

   ```bash
   sh PyPotteryLens_UNIX.sh
   ```
   This will install the required dependencies and start the application.

> ⚠️ **Important**: If you are encountering issues with the installation, run the cleanup script:
> ```bash
> sh cleanup.sh
> ```
> Then reinstall by running:
> ```bash
> sh PyPotteryLens_UNIX.sh
> ``` 

### Manual Way (Windows, Linux, MacOS)

1. Download the repository ZIP

2. Move into the downloaded folder

3. Install PyTorch

   - For CPU support:
   ```bash
   pip install torch torchvision torchaudio
   ```

   - For CUDA support (recommended for faster processing):

   Ensure you have a compatible NVIDIA GPU and the appropriate CUDA drivers installed: open the terminal and run the following command:

   ```bash
   nvidia-smi
   ```

   If you see a list of your GPU(s) and their status, you're good to go.

   Then, install PyTorch with CUDA support:

   ```bash
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```


4. Install the other required dependencies:
   ```bash
   pip install -r requirements.txt
   ```


5. Download the basic models from [HuggingFace](https://huggingface.co/lrncrd/PyPotteryLens/tree/main) and place them in to the `models_vision` (BasicModelv8_v01.pt) and `models_classifier` (model_classifier.pth).

   The project structure should look like this:

      ```
      pypotterylens/
      ├── app.py              # Main application file
      ├── utils.py            # Utility functions and processors
      ├── models.py           # Model definitions
      ├── models_vision/      # Detection models (create this folder and place the model here)
      ├── models_classifier/  # Classifier models (create this folder and place the model here)
      ├── outputs/            # Processing outputs
      ├── pdf2img_outputs/    # PDF conversion outputs
      ├── imgs/               # Application images
      └── requirements.txt    # Project dependencies
      ```

   The `models_vision` folder should look like this:

      ```
      models_vision/
      ├── BasicModelv8_v01.pt     # Main detection model
      └── [other_models].pt   # Additional detection models (if any)
      ```

   The `models_classifier` folder should look like this:

      ```
      models_classifier/
      ├── model_classifier.pth     # Main classifier model
      ```



## Let's get started

If you have installed the program using the quick installation, click on `PyPotteryLens_WIN.bat` on Windows or run `sh PyPotteryLens_UNIX.sh` in the bash on Linux or MacOS. The script detect the existing virtual environment and run the program. 

If you have installed the program manually, run the following command:

```bash
python app.py # python3 app.py for Linux and MacOS
```

This will open the GUI in your default browser.

## Usage

During the start-up, some useful information is displayed in the console. Including Directory Structure status, CUDA availability, main package versions.

Here's the description of a typical workflow:

### 1. Upload a PDF

In the "PDF document processing" tab, you'll start your pottery analysis journey by importing your PDF. The process begins with the `Click to Upload and process a File` button, which opens a file selector where you can choose any PDF from your computer. You'll find a helpful option called `Split scanned pages` - this checkbox is particularly useful when working with scanned books where each PDF page actually contains two physical pages side by side.


<p align="center">
<img src="imgs/tutorial_gif/example_pag_1.png" alt="One page layout" height="300"/>
</p>
<p align="center">
One page layout
</p>
</p>

<p align="center">
<img src="imgs/tutorial_gif/example_pag_2.png" alt="Two page layout" height="300"/>
<p align="center">
Two pages layout
</p>
</p>



> ⚠️ **Important**: When naming your PDF files, stick to a simple format without spaces or special characters. The recommended pattern is either `Author_Year.pdf` or `Context_Year.pdf`. For example:
> - `Cardarelli_2022.pdf`
> - `MonteCimino_2025.pdf`
> - `Veio_1963.pdf`
> This naming convention ensures smooth processing and helps maintain organized outputs.

After processing, you'll find your converted images in the `pdf2img_outputs` folder. The naming system is straightforward but differs depending on whether you used the split pages option:

For standard processing (Split pages unchecked):

- Files are named as `{Author_Year}_page_{PageNumber}.jpg`
- Example: `Cardarelli_2022_page_0.jpg` represents the first page of your document

For split processing (Split pages checked):

- Files are named as `{Author_Year}_page_{PageNumber}{side}.jpg`
- Example: `Cardarelli_2022_page_0a.jpg` represents the left side of the first page
- While `Cardarelli_2022_page_0b.jpg` represents the right side

<p align="center">
<img src="imgs/tutorial_gif/gif_1.gif" width="600"/>
</p>



### 2. Apply the Model

In the "Apply Model" tab, you can run the computer vision model on your processed images. In the `📁 Input Selection` box, you'll find two key elements: a folder selector where you can choose the images to process (these will be in your `pdf2img_outputs` folder), and a `Model` dropdown menu that lets you select from the available models in the `models_vision` folder.

The `⚙️ Model Parameter` box contains three important settings that help you fine-tune the detection process:

- `Confidence Threshold` (0.1 to 1.0): This determines how certain the model needs to be before identifying something as a pottery instance. A higher value (like 0.8) means the model will only detect pottery it's very confident about, reducing false positives but potentially missing some harder-to-detect pieces. A lower value (like 0.3) will catch more pottery instances but might include some incorrect detections.

- `Kernel Size`: This parameter affects how the segmentation masks are processed after detection. A larger kernel size creates smoother, more rounded edges but might lose some fine details. A smaller kernel preserves more detail but might leave the edges rougher. Think of it like adjusting the "brush size" for cleaning up the detection borders.

- `Iterations`: This determines how many times the post-processing is applied to the masks. More iterations create more pronounced effects - useful for closing small gaps or smoothing rough edges, but too many iterations might alter the pottery's shape too much. Work with this parameter alongside the Kernel Size to find the right balance.

Here is an example of different `Kernel Size` and `Iterations` applied to a pottery instance:

<p align="center">
<img src="imgs/tutorial_gif/parameter_comparison.png" width="500"/>
</p>

In the `🔧 Advanced Options` section, you'll find the `Diagnostic Mode` checkbox. When checked, it will only process the first 25 images in your folder, letting you quickly test your parameter settings without waiting for a full dataset to process. This is particularly helpful when you're fine-tuning the Model Parameters.

The right side of the page shows a gallery view of all the images waiting to be processed, helping you verify you've selected the correct folder and giving you a preview of what the model will work with.

When you're ready to process your images, click the `🚀 Apply Model` button. A progress bar will keep you informed as the model works through your images. The results are saved in a new folder called `{Author_Year}_mask` within the `outputs` directory, where you'll find your processed images with their segmentation masks visible.

<p align="center">
<img src="imgs/tutorial_gif/gif_2.gif" width="600"/>
</p>


### 3. Review annotations and extract istances

In the "Review Annotations and extract masks" tab, you can carefully review the computer vision results and extract individual pottery instances. In the `📁 Input Selection` area, you'll need to select the folder containing your processed images (you'll find this in the `outputs` folder with the name `{Author_Year}_mask`). 

Once you select a folder, a `FileExplorer` component appears, displaying all the images available for review. Clicking on any file's name opens it in the annotation revision tool, where you'll see the results of the automatic detection. If the model has successfully identified pottery instances, they'll appear as semi-transparent grey overlays on the image - this is the segmentation mask that defines each pottery item.

The annotation revision tool provides a simple but effective set of editing options. At the bottom of the tool, you'll find two main instruments:

- A **brush** (🖌️) tool for adding to the mask
- An **eraser** (🧹) tool for removing parts of the mask

Both tools come with adjustable size sliders, letting you fine-tune them for precise work. If you need to adjust your view of the whole interface, you can use the slider in the `📐 Editor Size` box to resize the annotation revision tool itself.

> 📝 **Note**: The current version of the brush tool doesn't support transparency, which means you won't be able to see the original image through your brush strokes while drawing. We're working on adding this feature in a future update to improve the editing experience.

Once you're satisfied with your annotations, the `📤 Extract Masks` button becomes your key to generating individual pottery instances. Clicking this button processes your annotations and saves the results in a new folder called `{Author_Year}_card` within the `outputs` directory. Each pottery instance gets its own image file, following a consistent naming pattern: `{Author_Year}_page_mask_layer_{InstanceNumber}.png`. 

For instance, if you're working with a document called `Cardarelli_2022.pdf`, the first extracted instance would be saved as `Cardarelli_2022_page_mask_layer_0.jpg`.

> 💡 **Tip**: It's good practice to review a few extracted instances after processing to ensure the masks are clean and accurate. This can save time in later processing stages.


<p align="center">
<img src="imgs/tutorial_gif/gif_3.gif" width="600"/>
</p>

### 4. Tabular information

In the "Tabular Information" tab, you can select the folder containing the images processed in the previous step (in the `outputs` folder: `{Author_Year}_card`). On the left side of the page, you'll see an Annotated Image component that displays both the original image and highlighted bounding boxes around each pottery instance. 

Navigation between images is straightforward using the Previous and Next buttons, allowing you to move through your dataset efficiently.
At the bottom of the page, you'll find controls that let you focus on specific instances - when you select an instance, the Annotated Image component will automatically highlight just that piece, making it easier to verify details. This feature is particularly helpful when working with pages containing multiple pottery items.

On the right side of the page, you'll find the heart of the data management system - an interactive table component. When you first start, this table will contain just the basic ID for each instance on the page, but it's fully customizable to your needs. Adding new columns is simple: just type the desired column name in the "New Column Name" text box and click the Add Column button. The new column appears immediately in your table, ready for data entry.
Data entry is straightforward - simply click on any cell in the table and type your value. One of the most convenient features is that every change you make is saved automatically, so you never have to worry about losing your work. All this information is stored in a CSV file called `mask_info.csv`, which you can find in the outputs folder.


> ⚠️ **Important Note**: When entering data, remember to avoid using commas in your values, as this can cause issues with the CSV file format. If you need to work with complex data or make bulk edits, you can always open the CSV file in Excel or Google Sheets.


<p align="center">
<img src="imgs/tutorial_gif/gif_4.gif" width="600"/>
</p>

### 5. Post Processing

In the "Post Processing" tab, you can select the folder containing the images processed in the previous step (in the `outputs` folder: `{Author_Year}_card`). This tab is designed to help you standardize the orientation and classification of your pottery instances. In the `Processing Options` box, you'll find two powerful automatic processing options:

- `Auto Vertical Flip`: This feature automatically detects and corrects instances where the pot's mouth is facing downward, flipping them to maintain a consistent upward orientation.
- `Auto Horizontal Flip`: Similarly, this option ensures all pottery profiles are oriented to the left side of the image, maintaining consistency across your dataset.

The Action box contains several important tools. The `Process All Images` button applies your selected processing options to every image in the folder at once - a real time-saver when working with large collections. On the right side of the page, you can compare the original and processed versions of each image side by side, making it easy to verify the results. Navigation buttons let you move through your collection smoothly.

Sometimes the automatic processing might not get things quite right - that's why we've included manual correction tools. If you notice any orientation issues, you can easily fix them using the `Flip Vertical` and `Flip Horizontal` buttons. You can also update the classification of each pot between complete (ENT) and fragmentary (FRAG) using the `Type` dropdown menu. Don't worry about saving - every change you make is automatically stored.

All processed images are saved in a new folder called `{Author_Year}_transformed_card` within the `outputs` folder. Once you're happy with the processing, you can use the `Merge Annotations` button in the Action box to combine your type classifications with the tabular information from your `mask_info.csv` file. This creates a comprehensive record in a new file called `merged_annotations.csv`.

The final step is exporting your results, which you can initiate by clicking the `📦 Export Results` button. This opens the `📦 Export Options` dialog, where you can customize your export. One key feature is the ability to assign an acronym to your dataset - for example, using "CRD" would name your files sequentially as `CRD_0.png`, `CRD_1.png`, and so on. For documentation purposes, you can also generate a PDF catalog - just check the Generate PDF Catalog box, select your preferred format (A4, A3, A5, Letter, or Legal), and adjust the Image Scale Factor to ensure your pottery fits perfectly on the page. All exported files are saved in a new folder named after your chosen acronym within the `outputs` directory.

<p align="center">
<img src="imgs/tutorial_gif/gif_5.gif" width="600"/>
</p>

### 6. Advanced Features (New!)

The "Advanced Features" tab provides professional tools for archaeological research:

#### 🎨 **Color Correction**
- Normalize colors across your entire dataset
- Auto white balance and histogram equalization
- Preview corrections before applying

#### 🔍 **Comparison View**
- Compare multiple pottery items side-by-side
- Choose horizontal or vertical layouts
- Add labels for easy identification

#### 🗺️ **GIS Export**
- Export to GeoJSON for QGIS/ArcGIS integration
- Include full metadata and measurements
- Support for spatial analysis workflows

#### 🏛️ **CIDOC-CRM Export**
- Export following international archaeological standards
- Choose between RDF/XML and JSON-LD formats
- Include morphometric measurements and context

#### 🏷️ **Metadata Management**
- Add hierarchical tags (typology, decoration, condition)
- Define relationships between pottery items
- Track complete provenance history

#### 📊 **Statistical Analysis**
- Generate comprehensive dashboards
- Automatic morphometric measurements
- Clustering analysis to group similar items
- Export publication-ready visualizations

#### 📄 **Multi-format Reports**
- Generate reports in PDF, DOCX, or HTML
- Support for English, Italian, and Spanish
- Include images, statistics, and bibliography
- Professional templates for different uses

### API Usage

To enable REST API access to your data:

```python
# In a separate terminal after starting PyPotteryLens
from api_reports import PotteryAPI
api = PotteryAPI()
api.run(host='0.0.0.0', port=5000)
```

API endpoints will be available at `http://localhost:5000/api/`

## Dependencies

Core dependencies:
- **gradio**: Web-based user interface
- **torch**: Deep learning framework (with CUDA/MPS support)
- **ultralytics**: YOLO object detection
- **PyMuPDF**: PDF processing
- **opencv-python**: Computer vision operations

Analysis & visualization:
- **scikit-learn**: Machine learning and clustering
- **scikit-image**: Image processing
- **matplotlib** & **seaborn**: Statistical visualizations
- **pandas**: Data manipulation

Export & reporting:
- **reportlab**: PDF generation
- **python-docx**: DOCX generation
- **flask** & **flask-restful**: REST API
- **jinja2**: Template engine

Full list available in `requirements.txt`

## Version History

- 0.1.4 (Current Development)
   - Added Advanced Features tab with professional archaeological tools
   - Implemented CIDOC-CRM export for international standard compliance
   - Added SQLite database replacing CSV files
   - Integrated color normalization and image enhancement tools
   - Added hierarchical metadata management with provenance tracking
   - Implemented statistical analysis with morphometric measurements
   - Added multi-format, multi-language report generation
   - Created REST API for external system integration
   - Added clustering analysis for pottery grouping
   - Implemented GIS export (GeoJSON) for spatial analysis
   - Enhanced MPS detection for Apple Silicon Macs
   - Fixed Python 3.13 compatibility issues
   - Added dark mode theme
   - Added comparison view for side-by-side analysis
   - Implemented smart caching for performance optimization
   - Added cleanup script for easier troubleshooting

- 0.1.3
   - Minor bug fixes

- 0.1.2
   - Checked compatibility with MacOS (Sonoma 15.2)
   - Added MPS support for Apple Silicon Device
   - Added a white border to extracted images

- 0.1.1
   - Checked compatibility with Linux (Ubuntu 24.10) and MacOS (Sonoma 14)
   - Improved bin packing algorithm for PDF creation
   - Added paper's (PyPotteryLens: An Open-Source Deep Learning Framework for Automated Digitisation of Archaeological Pottery Documentation) supporting scripts
   - Reworked the GUI for better user experience and compatibility
- 0.1.0
  - Initial Release

## System compatibility

- Windows 11
- Ubuntu 24.10
- MacOS Sonoma 14
- MacOS Sequoia 15.2

## Hardware Support

### GPU Acceleration
PyPotteryLens supports multiple hardware acceleration options:
- **NVIDIA GPUs**: CUDA support for NVIDIA graphics cards
- **Apple Silicon**: MPS (Metal Performance Shaders) support for M1/M2/M3 Macs
- **CPU**: Fallback for systems without GPU acceleration

The application automatically detects and uses the best available option.

## Known Issues ⚠️

- The brush tool in the annotation revision tool doesn't support transparency, making it difficult to see the original image through brush strokes. This feature is planned for a future update.
- For older version of MacOS (<= Monterey 12.7.5), the last version of Pytorch supported is `2.2.2`. Please modify the `requirements.txt` file accordingly.
- Python 3.13 compatibility: If using Python 3.13, ensure scikit-image version is 0.25.0 or higher.


## Contributors

- Lorenzo Cardarelli
