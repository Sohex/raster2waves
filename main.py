from io import BytesIO
import numpy as np
from pathlib import Path
from PIL import Image, ImageTk
from skimage.measure import block_reduce
import tkinter as tk
from tkinter import filedialog, messagebox
import os
os.environ['path'] += r';C:\Program Files\Inkscape\bin'
import cairocffi as cairo
import cairosvg
from concurrent.futures import ThreadPoolExecutor

class RasterToWavesDriver:
    """
    A class that converts raster images into waveforms.

    Args:
        channels (dict): A dictionary containing the color channels of the image.
        aspect_ratio (float): The aspect ratio of the canvas.

    Attributes:
        channels (dict): A dictionary containing the color channels of the image.
        aspect_ratio (float): The aspect ratio of the canvas.
        waves (dict): A dictionary to store the calculated waves for each color channel.
        svgs (dict): A dictionary to store the rendered SVGs for each color channel.
    """

    def __init__(self, channels, aspect_ratio) -> None:
        """
        Initializes a new instance of the class.

        Args:
            channels (int): The number of channels.
            aspect_ratio (float): The aspect ratio.

        Returns:
            None
        """
        self.channels = channels
        self.aspect_ratio = aspect_ratio
        self.waves = {}
        self.svgs = {}

    def process_channel(self, color, downsample_factor, contrast_cutoff=1, **kwargs):
        """
        Process a channel of the image.

        Args:
            color (str): The color channel to process.
            downsample_factor (int): The factor by which to downsample the channel.
            contrast_cutoff (float, optional): The percentage of contrast to cut off. Defaults to 1.
            **kwargs: Additional keyword arguments.

        Returns:
            None

        """
        channel_data = self.channels[color]
        minval, maxval = np.percentile(channel_data, [contrast_cutoff, 100 - contrast_cutoff])
        channel_with_contrast = np.clip(channel_data, minval, maxval)
        if maxval != minval:
            channel_with_contrast = np.divide(np.subtract(channel_with_contrast, minval), maxval - minval)
        else:
            channel_with_contrast = np.zeros_like(channel_data)

        self.channels[color] = block_reduce(channel_with_contrast, block_size=downsample_factor, func=np.mean)

    @staticmethod
    def phase_reconciliation(wave_numbers):
        """
        Reconciles the phase shift for a given list of wave numbers.

        Parameters:
        wave_numbers (list): A list of wave numbers.

        Returns:
        numpy.ndarray: An array containing the reconciled phase shift values.
        """
        phase_shift = [0]
        for w in wave_numbers[:-1]:
            if phase_shift[-1]:
                phase_shift.append(1 - (w % 2))
            else:
                phase_shift.append(w % 2)
        return np.array(phase_shift)

    def calculate_waves(self, color, max_wave_num=4, wave_num_factor=1.2, max_amplitude=8, amplitude_factor=1, resolution=20, **kwargs):
        """
        Calculate waves for a given color channel.

        Args:
            color (str): The color channel to calculate waves for.
            max_wave_num (int, optional): The maximum number of waves. Defaults to 4.
            wave_num_factor (float, optional): The factor to adjust the number of waves. Defaults to 1.2.
            max_amplitude (int, optional): The maximum amplitude of the waves. Defaults to 8.
            amplitude_factor (int, optional): The factor to adjust the amplitude of the waves. Defaults to 1.
            resolution (int, optional): The number of points to calculate along the x-axis. Defaults to 20.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        wave_num = (self.channels[color] ** (1 / wave_num_factor) * max_wave_num).astype(int)
        phase_shift = np.apply_along_axis(self.phase_reconciliation, 1, wave_num)

        amplitude = self.channels[color] ** (1 / amplitude_factor) * max_amplitude
        get_wave = lambda x: np.sin(x * wave_num * np.pi + (phase_shift * np.pi)) * amplitude

        x = np.linspace(0, 1, resolution, endpoint=False)
        waves_separated = np.array(list(map(get_wave, x)))
        waves = waves_separated.swapaxes(0, 1).reshape(waves_separated.shape[1], -1, order='F')

        self.waves[color] = waves

    def render_waves(self, channel='K', line_color=(0, 0, 0), canvas_width_in=8.5, canvas_height_in=11, border_in=.5, line_width=.2, **kwargs):
        if self.waves is None:
            raise ValueError("Calculate waves before rendering!")
        POINTS_PER_INCH = 72
        canvas_width_points, canvas_height_points, border_points = canvas_width_in * POINTS_PER_INCH, canvas_height_in * POINTS_PER_INCH, border_in * POINTS_PER_INCH
        max_width, max_height = canvas_width_points - 2 * border_points, canvas_height_points - 2 * border_points
        draw_width, draw_height = (self.aspect_ratio * max_height, max_height) if self.aspect_ratio < 1 else (max_width, max_width / self.aspect_ratio)
        horizontal_border, vertical_border = (canvas_width_points - draw_width) / 2, (canvas_height_points - draw_height) / 2
        line_spacing = draw_height / self.waves[channel].shape[0]
        vertical_offset = (np.ones(self.waves[channel].shape) * line_spacing).cumsum(axis=0) - line_spacing
        vertical_offset += vertical_border
        wave_offset = self.waves[channel] + vertical_offset
        x_values = np.linspace(horizontal_border, draw_width + horizontal_border, self.waves[channel].shape[1])
        svg = BytesIO()
        canvas = cairo.SVGSurface(svg, canvas_width_points, canvas_height_points)
        ctx = cairo.Context(canvas)
        for line in wave_offset:
            ctx.move_to(x_values[0], line[0])
            for i, value in enumerate(line):
                ctx.line_to(x_values[i], value)
        ctx.set_line_width(line_width)
        ctx.set_source_rgb(*line_color)
        ctx.stroke()
        canvas.finish()
        self.svgs[channel] = svg

    def save_svg(self, svg, outfile):
        """
        Save an SVG object to a file.

        Args:
            svg (SVG): The SVG object to be saved.
            outfile (str): The path to the output file.

        Returns:
            None
        """
        with open(outfile, "wb") as out:
            out.write(svg.getbuffer())

    def process_image(self, image_prefix, downsample_factor, canvas_width=8.5, canvas_height=11, border_width=0.5, line_width=0.25, **kwargs):
        """
        Process the image by performing the following steps:
        1. Process each color channel.
        2. Calculate waves for each color channel.
        3. Render the waves on a canvas.

        Args:
            image_prefix (str): The prefix of the image file.
            downsample_factor (float): The factor by which to downsample the image.
            canvas_width (float, optional): The width of the canvas in inches. Defaults to 8.5.
            canvas_height (float, optional): The height of the canvas in inches. Defaults to 11.
            border_width (float, optional): The width of the border around the canvas in inches. Defaults to 0.5.
            line_width (float, optional): The width of the lines used to render the waves in inches. Defaults to 0.25.
            **kwargs: Additional keyword arguments for calculating waves.

        Returns:
            None
        """
        channel_opts = [
            {"color": "C", "line_color": (0, 255, 255)},
            {"color": "M", "line_color": (255, 0, 255)},
            {"color": "Y", "line_color": (255, 255, 0)},
            {"color": "K", "line_color": (0, 0, 0)},
        ]

        def process_channel_and_calculate_waves(channel):
            self.process_channel(channel["color"], downsample_factor)
            self.calculate_waves(channel["color"], **kwargs)
            self.render_waves(channel["color"], channel["line_color"], canvas_width, canvas_height, border_width, line_width)

        with ThreadPoolExecutor() as executor:
            executor.map(process_channel_and_calculate_waves, channel_opts)

def split_cmyk(rgb_array, threshold=1):
    """
    Splits an RGB image into CMYK channels.

    Parameters:
    - rgb_array (numpy.ndarray): The input RGB image as a numpy array.
    - threshold (float): The threshold value for channel separation. Default is 1.

    Returns:
    - channels (dict): A dictionary containing the CMYK channels as numpy arrays.
                       The keys are 'C', 'M', 'Y', and 'K'.
    """

    data = rgb_array.astype(float) / 255
    threshold = threshold / 255
    
    channel_max = data.max(2)
    channel_max[channel_max < threshold] = threshold
    
    k = 1 - channel_max
    c = (1 - data[:, :, 0] - k) / channel_max
    m = (1 - data[:, :, 1] - k) / channel_max
    y = (1 - data[:, :, 2] - k) / channel_max
    
    result = 1 - np.array([c, m, y, k])
    channels = {}
    channels["C"], channels["M"], channels["Y"], channels["K"] = [(_ * 255).round().astype(np.uint8) for _ in result]
    return channels

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip, text=self.text, background="#ffffe0", relief="solid", borderwidth=1)
        label.pack(ipadx=1)

    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

class RasterToWavesGUI(tk.Tk):
    """
    A GUI application for converting raster images to waves.

    Attributes:
        driver: The driver used for image processing.
        image_path: The path of the loaded image.
        frame_controls: The frame containing the control buttons.
        load_button: The button for loading an image.
        save_button: The button for saving SVGs.
        channel_var: The variable for storing the selected channel.
        radio_frame: The frame containing the radio buttons for channel selection.
        param_frame: The frame containing the parameter labels and entry fields.
        params: A dictionary of parameter names and their default values, variable types, and value ranges.
        canvas: The canvas for displaying the converted image.
        image_label: The label for displaying the loaded image.
        svg_image: The converted SVG image.
        svg_label: The label for displaying the SVG image.
        current_image: The currently loaded image.
    """

    def __init__(self):
        super().__init__()
        self.title('Raster to Waves Converter')
        self.geometry('1280x720')
        self.driver = None
        self.image_path = None
        self.init_ui()
        
    def init_ui(self):
        """
        Initializes the user interface by creating and configuring the necessary widgets and variables.
        
        This method sets up the frame controls, load and save buttons, radio buttons for channel selection,
        parameter frame for setting values, canvas for displaying images, and initializes various variables
        and bindings.
        """
        self.frame_controls = tk.Frame(self)
        self.frame_controls.pack(fill=tk.X, side=tk.TOP)
        
        self.load_button = tk.Button(self.frame_controls, text='Load Image', command=self.load_image)
        self.load_button.pack(side=tk.LEFT)
        
        self.save_button = tk.Button(self.frame_controls, text='Save SVGs', command=self.save_svgs, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT)
        
        self.channel_var = tk.StringVar(self)
        self.channel_var.set("All")
        
        self.radio_frame = tk.Frame(self.frame_controls)
        self.radio_frame.pack(side=tk.RIGHT)
        
        channels = ["C", "M", "Y", "K", "All"]
        for channel in channels:
            rb = tk.Radiobutton(self.radio_frame, text=channel, variable=self.channel_var, value=channel, command=self.switch_channel)
            rb.pack(side=tk.LEFT)
              
        self.param_frame = tk.Frame(self)
        self.param_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=10)
        
        # Define default values and update methods
        self.params = {
            "downsample_factor": (12, tk.IntVar, 0, 100, "Factor by which to downsample the image."),
            "contrast_cutoff": (1, tk.IntVar, 0, 255, "Adjust contrast cutoff. Lower values increase contrast."),
            "max_wave_num": (4, tk.IntVar, 0, 255, "Maximum number of waves per half period."),
            "wave_num_factor": (1.2, tk.DoubleVar, 0., 10., "Weighting factor for wave number calculation."),
            "max_amplitude": (8, tk.DoubleVar, 0., 255., "Maximum amplitude of the waves."),
            "amplitude_factor": (1, tk.DoubleVar, 0., 10., "Weighting factor for amplitude calculation."),
            "resolution": (20, tk.IntVar, 0, 255, "Number of points to calculate along the x-axis."),
            "canvas_width": (8.5, tk.DoubleVar, 0., 100., "Width of the canvas in inches."),
            "canvas_height": (11, tk.DoubleVar, 0., 100., "Height of the canvas in inches."),
            "border": (.5, tk.DoubleVar, 0., 100., "Width of the border around the canvas in inches."),
            "line_width": (.2, tk.DoubleVar, 0., 10., "Width of the lines used to render the waves.")
        }

        # Create label and entry pairs
        for i, (param, (default_value, var_type, min_value, max_value, text)) in enumerate(self.params.items()):
            row = i // 4
            column = i % 4
            var = var_type()
            var.set(default_value)
            
            # Convert the parameter name to a more readable format
            param_display_name = param.replace('_', ' ').title() + ":"
            
            label, entry = self.create_label_and_entry(row, column, param_display_name, var, self.update_value)
            setattr(self, f"{param}_label", label)
            setattr(self, f"{param}_entry", entry)
            setattr(self, f"{param}_var", var)
            
            tooltip_text = f"{text}\nMin value: {min_value}, Max value: {max_value}"
            ToolTip(entry, tooltip_text)

        self.canvas = tk.Canvas(self, bg='grey')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self.on_resize_debounced)
        self.__resize_id = None
        
        self.image_label = None
        self.svg_image = None
        self.svg_label = None
        self.current_image = None
        self.img_rgb = None
        self.aspect_ratio = None
        self.driver = None
    
    def create_label_and_entry(self, row, column, text, var, update_method):
        """
        Create a label and entry widget in the parameter frame.

        Args:
            row (int): The row index for the grid layout.
            column (int): The column index for the grid layout.
            text (str): The text to display on the label.
            var (tk.StringVar): The variable to associate with the entry widget.
            update_method (function): The method to call when the entry is updated.

        Returns:
            tuple: A tuple containing the label and entry widgets.

        """
        label = tk.Label(self.param_frame, text=text)
        label.grid(row=row, column=column*2, padx=5, pady=5)
        entry = tk.Entry(self.param_frame, width=4, textvariable=var)
        entry.grid(row=row, column=column*2+1, padx=5, pady=5)
        entry.bind("<Return>", update_method)
        return label, entry
    
    def update_value(self, event=None):
        """
        Update the values of the parameters based on user input.

        Args:
            event (Event, optional): The event that triggered the update. Defaults to None.
        """
        for param, (default_value, var_type, min_value, max_value, text) in self.params.items():
            entry = getattr(self, f"{param}_entry")
            var = getattr(self, f"{param}_var")
            try:
                value = var.get()
                if param == "border":
                    if not 0 <= value < min(self.canvas_width_var.get(), self.canvas_height_var.get()):
                        raise ValueError
                elif not min_value < value <= max_value:
                    raise ValueError
                self.process_image()
            except ValueError:
                var.set(default_value)
                if param == "border":
                    messagebox.showwarning("Warning", f"{param.replace('_', ' ').title()} must be {var_type.__name__} in the range [{min_value}, {min(self.canvas_width_var.get(), self.canvas_height_var.get())}).")
                else:
                    messagebox.showwarning("Warning", f"{param.replace('_', ' ').title()} must be {var_type.__name__} in the range ({min_value}, {max_value}].")
    
    def on_resize_debounced(self, event):
        """
        Debounces the on_resize event by canceling any previous resize event and scheduling a new one after 200 milliseconds.

        Args:
            event: The resize event that triggered the method.
        """
        if self.__resize_id:
            self.after_cancel(self.__resize_id)
        self.__resize_id = self.after(200, self.on_resize, event)
    
    def on_resize(self, event):
        """
        Event handler for the resize event.
        Parameters:
        - event: The resize event object.
        Description:
        - This method is called when the window is resized. It checks if an image path is set and then
            calls the display_image method to display the image and the switch_channel method to switch the channel.
        """
        if self.image_path:
            self.display_image()
            self.switch_channel()
    
    def load_image(self):
            """
            Opens a file dialog to select an image file and processes the selected image.

            Returns:
                None
            """
            file_path = filedialog.askopenfilename(filetypes=[("JPEG files", "*.jpg"), ("JPEG files", "*.jpeg")])
            if not file_path:
                return

            max_dim = 1024
            self.image_path = file_path
            try:
                with Image.open(file_path) as img:
                    img_rgb = np.asarray(img.convert("RGB"))
                    if max(img_rgb.shape) > max_dim:
                        raise ValueError(f"Image dimensions must be less than {max_dim} pixels.")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred while loading the image: {e}")
                self.image_path = None
                return
            
            self.img_rgb = img_rgb
            self.aspect_ratio = img_rgb.shape[1] / img_rgb.shape[0]
                
            self.process_image()

    def process_image(self):
        """
        Process the given image file.

        Returns:
            None
        """
        params = {
            "downsample_factor": self.downsample_factor_var.get(),
            "contrast_cutoff": self.contrast_cutoff_var.get(),
            "max_wave_num": self.max_wave_num_var.get(),
            "wave_num_factor": self.wave_num_factor_var.get(),
            "max_amplitude": self.max_amplitude_var.get(),
            "amplitude_factor": self.amplitude_factor_var.get(),
            "resolution": self.resolution_var.get(),
            "canvas_width": self.canvas_width_var.get(),
            "canvas_height": self.canvas_height_var.get(),
            "border": self.border_var.get(),
            "line_width": self.line_width_var.get()
        }

        split_img = split_cmyk(self.img_rgb)
        self.aspect_ratio = self.img_rgb.shape[1] / self.img_rgb.shape[0]

        self.driver = RasterToWavesDriver(split_img, self.aspect_ratio)
        self.driver.process_image(Path(self.image_path).stem, **params)

        self.display_image()
        self.save_button['state'] = tk.NORMAL
        
    def display_image(self):
        """
        Display an image on the canvas.

        Returns:
            None
        """
        new_width = self.canvas.winfo_width() // 2
        new_height = self.canvas.winfo_height()
        
        img = Image.fromarray(self.img_rgb)
        img_new_width, img_new_height = self.calculate_new_size(new_width, new_height, self.aspect_ratio)
        
        img = img.resize((img_new_width, img_new_height), Image.LANCZOS)
        self.current_image = ImageTk.PhotoImage(img)
        
        if self.image_label is None:
            self.image_label = tk.Label(self.canvas, image=self.current_image)
            self.image_label.pack(side=tk.LEFT, padx=10, pady=10)
        else:
            self.image_label.config(image=self.current_image)
            
        self.switch_channel()
        
    def calculate_new_size(self, new_width, new_height, aspect_ratio):
        """
        Calculates the new width and height based on the given aspect ratio.
        Parameters:
        - new_width (int): The desired new width.
        - new_height (int): The desired new height.
        - aspect_ratio (float): The aspect ratio to maintain.
        Returns:
        - tuple: A tuple containing the new width and height.
        """
        if new_width / new_height > aspect_ratio:
            new_width = int(new_height * aspect_ratio)
        else:
            new_height = int(new_width / aspect_ratio)
            
        return new_width, new_height
        
    def switch_channel(self):
        """
        Switches the channel and updates the displayed SVG image.
    
        Returns:
            None
        """
        canvas_width = self.canvas.winfo_width() // 2
        canvas_height = self.canvas.winfo_height()
        white_bg = Image.new('RGBA', (canvas_width, canvas_height), 'white')
        channels = self.driver.svgs.keys() if self.channel_var.get() == "All" else [self.channel_var.get()]
    
        for channel in channels:
            if self.driver and channel in self.driver.svgs:
                svg_data = self.driver.svgs[channel].getvalue().decode("utf-8")
    
                png_data = cairosvg.svg2png(bytestring=svg_data)
                image = Image.open(BytesIO(png_data))
    
                image_new_width, image_new_height = self.calculate_new_size(canvas_width, canvas_height, self.aspect_ratio)
    
                # Check if the new image size is larger than the canvas size
                if image_new_width > canvas_width or image_new_height > canvas_height:
                    # Adjust the image size to fit within the canvas
                    image_new_width = min(image_new_width, canvas_width)
                    image_new_height = min(image_new_height, canvas_height)
    
                image = image.resize((image_new_width, image_new_height), Image.LANCZOS)
    
                # Calculate the offset after adjusting the image size
                offset_x = (canvas_width - image_new_width) // 2
                offset_y = (canvas_height - image_new_height) // 2
    
                white_bg.paste(image, (offset_x, offset_y), image)
    
        self.svg_image = ImageTk.PhotoImage(white_bg)
    
        if self.svg_label is None:
            self.svg_label = tk.Label(self.canvas, image=self.svg_image, width=canvas_width, height=canvas_height)
            self.svg_label.pack(side=tk.RIGHT, padx=10, pady=10)
        else:
            self.svg_label.config(image=self.svg_image, width=canvas_width, height=canvas_height)
                
    def save_svgs(self):
            """
            Saves the SVGs generated by the driver to the specified folder.

            If there are no SVGs to save or if no folder is selected, the method returns early.

            Args:
                None

            Returns:
                None
            """
            if not self.driver or not self.driver.svgs:
                messagebox.showerror("Error", "No SVGs to save. Please load and process an image first.")
                return
                
            folder_path = filedialog.askdirectory()
            if not folder_path:
                return
                
            for color, svg in self.driver.svgs.items():
                file_path = os.path.join(Path(folder_path), f"{Path(self.image_path).stem}_{color}.svg")
                self.driver.save_svg(svg, file_path)
                
            messagebox.showinfo("Success", "SVGs have been saved successfully.")
        
if __name__ == "__main__":
    app = RasterToWavesGUI()
    app.mainloop()

