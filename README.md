# Raster to Waves Converter

## Overview

This program converts raster images into waveforms, allowing users to visualize image data in a unique and artistic way. It processes images by splitting them into their CMYK color channels, applying various transformations to generate waveforms from the image data, and then rendering these waveforms as SVGs.

## Installation

### Prerequisites

- Python3
- libcairo (on Linux this comes from your package manager, on Windows we're using the one provided by inkscape by default, but anything that packages libcairo2.dll will work.)

### Steps

1. Ensure that Python and pip are installed on your system.

2. Clone the repository or download the program files to your local machine.

3. Install the required Python packages:

```bash
pip install -r requirements.txt
```

4. Run it!
```
python main.py
```

## Usage

### Starting the Program

Run the program using Python:

```bash
python main.py
```

Alternatively use the packaged pyinstaller executable, e.g. raster2waves.exe

### User Interface

The GUI allows you to:

- Load raster images for conversion.
- Adjust parameters for the conversion process, such as downsample factor, contrast cutoff, maximum wave number, and others.
- Save the generated SVG files.

### Main Features

- **Image Loading**: Use the "Load Image" button to select and load a raster image into the program. N.B. Images should have a strong contrast between foreground and background and the background should be black, not white.
- **Parameter Adjustment**: Fine-tune the conversion process using the provided parameters in the GUI. Hover over each parameter entry for a tooltip explaining its function.
- **Channel Selection**: Choose between the CMYK color channels or select "All" to view all channels simultaneously.
- **SVG Rendering and Saving**: After processing, the waveforms are displayed in the GUI. Use the "Save SVGs" button to save the rendered SVGs to your local system.

### Advanced Usage

For advanced users, the program's modular design allows for customization and integration into other Python projects. The `RasterToWavesDriver` class can be used independently of the GUI for script-based processing.

## Contributing

Contributions to the project are welcome. Please follow the standard fork and pull request workflow.

## License

This project is licensed under the GPLv3 License - see the LICENSE file for details.

## Acknowledgments

This project utilizes several open-source software libraries, including NumPy, Pillow, scikit-image, CairoCFFI, and CairoSVG. It is derived from the SquiggleDraw and PySquiggleDraw projects. Thanks to the developers of these libraries and projects for their invaluable contributions to the open-source community.
