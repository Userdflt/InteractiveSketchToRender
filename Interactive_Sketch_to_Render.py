import threading
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageDraw, ImageOps, ImageTk
import numpy as np
import torch

# Diffusers imports
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLInpaintPipeline,
    ControlNetModel,
    AutoencoderKL,
    DDIMScheduler,
    UniPCMultistepScheduler
)
from diffusers.utils import load_image

##############################################
# Global Pipeline Initialization and Configs
##############################################

# CONFIG
# MODEL CONFIG
CONTROLNET_MODEL = "xinsir/controlnet-scribble-sdxl-1.0"
SDXL_MODEL = "SG161222/RealVisXL_V4.0_Lightning"
# INFERENCE CONFIG
CONTROLNET_CONDITIONING_SCALE = 0.7  
NUM_INFER_STEPS = 15
INPAINT_NUM_INFER_STEPS = 20
INPAINT_STRENGTH = 0.99
GUIDANCE_SCALE = 3
INPAINT_GUIDANCE_SCALE = 8
GUESS_MODE = True
ETA = 0.5
WIDTH = 512
HEIGHT = 512

# Load the ControlNet model (using FP16 precision)
controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL,
    torch_dtype=torch.float16,
    use_safetensors=True,
)

# Load the main SDXL pipeline with ControlNet attached
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    SDXL_MODEL,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    use_safetensors=True,
    # variant="fp16", ## enable when using RealVisXL 5.0 Lightning
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

inpaint_pipe = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
inpaint_pipe.to("cuda")

##############################################
# Inference Helper Functions
##############################################

def generate_preview_image(sketch_image, prompt, neg_prompt):
    """
    Generate a preview (low-resolution) image from the composite sketch.
    The input is resized to 512x512 and a control image is created from a grayscale/inverted version.
    """
    # Resize input to 512x512
    input_image = sketch_image.resize((WIDTH, HEIGHT))
    # Create control image (line art effect)
    control_image = ImageOps.invert(ImageOps.grayscale(sketch_image))
    
    low_res_img = pipe(
        prompt,
        negative_prompt=neg_prompt,
        image=input_image,
        control_image=control_image,
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=NUM_INFER_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
        guess_mode=GUESS_MODE,
        eta=ETA,
        dynamic_threshold=True
        # vae=vae
    ).images[0]
    
    return low_res_img

def generate_inpaint_image(base_image, mask_image, prompt, neg_prompt):
    """
    Generate an inpainted image by replacing the masked areas in the base_image with generated content.
    The mask_image should be in mode "L" where white (value >128) indicates areas to inpaint.
    """
    # Create an inpainting input by replacing masked areas with white
    base_np = np.array(base_image)
    mask_np = np.array(mask_image.convert("L"))
    threshold = 128
    inpaint_np = np.where(mask_np[:, :, None] > threshold, 255, base_np)
    inpaint_input = Image.fromarray(inpaint_np.astype(np.uint8))
    
    result_img = inpaint_pipe(
        prompt,
        # negative_prompt=neg_prompt,
        image=base_image,
        mask_image=inpaint_input,  # using the inpaint input as control; adjust as needed
        width=WIDTH,
        height=HEIGHT,
        strength=INPAINT_STRENGTH,
        num_inference_steps=INPAINT_NUM_INFER_STEPS,
        guidance_scale=INPAINT_GUIDANCE_SCALE,
    ).images[0]
    
    # Composite the generated result with the original base image using the mask
    mask_binary = mask_image.convert("L").point(lambda p: 255 if p > threshold else 0)
    final_image = Image.composite(result_img, base_image, mask_binary)
    return final_image

##############################################
# Sketch Application with Inference, Inpainting & UI
##############################################

class SketchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sketch & Inpaint Window")
        
        # Set up two canvases in a single row (drawing canvas left, generated output right)
        self.canvas_width = WIDTH
        self.canvas_height = HEIGHT
        
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)
        
        blank_image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.blank_preview = ImageTk.PhotoImage(blank_image)
        self.output_label = tk.Label(root, image=self.blank_preview, bg="white")
        self.output_label.grid(row=0, column=1, padx=10, pady=10)
        
        # Bind mouse motion to update the brush preview circle
        self.canvas.bind("<Motion>", self.update_cursor_circle)
        self.eraser_cursor_id = None
        
        # PIL image for drawing (starts as blank white)
        self.sketch = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.sketch)
        self.undo_stack = []
        self.current_stroke_item_ids = []
        
        # Default drawing settings
        self.current_color = "black"  # "black" for drawing; "white" for eraser
        self.brush_width = 3
        
        # Inpainting mode flag and mask (for inpainting strokes)
        self.inpainting_mode = False
        self.inpaint_mask = None
        self.base_for_inpainting = None  # The base image to inpaint on
        
        # Bind mouse events for drawing
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.root.bind("<Control-z>", self.on_ctrl_z)
        
        # ---- Control Buttons Layout ----
        # Row 1: Inference and Clear Canvas buttons
        self.inference_button = tk.Button(root, text="Inference", command=self.trigger_inference)
        self.inference_button.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        self.clear_button = tk.Button(root, text="Clear Canvas", command=self.clear_canvas)
        self.clear_button.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        
        # Row 2: Toggle Eraser and Toggle Inpainting Mode buttons
        self.eraser_button = tk.Button(root, text="Toggle Eraser", command=self.toggle_eraser)
        self.eraser_button.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        
        self.toggle_inpaint_mode_button = tk.Button(root, text="Enter Inpainting Mode", command=self.toggle_inpainting_mode)
        self.toggle_inpaint_mode_button.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        
        # Row 3: Inpaint and Save Generated buttons
        self.inpaint_button = tk.Button(root, text="Apply Inpainting", command=self.trigger_inpaint_process)
        self.inpaint_button.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        self.inpaint_button.config(state=tk.DISABLED)
        
        self.save_button = tk.Button(root, text="Save Generated", command=self.save_generated)
        self.save_button.grid(row=3, column=1, padx=10, pady=5, sticky="ew")
        
        # Row 4: Brush size slider (spanning both columns)
        self.brush_size_scale = tk.Scale(root, from_=1, to=50, orient=tk.HORIZONTAL,
                                         label="Brush/Eraser Size", command=self.update_brush_size)
        self.brush_size_scale.set(self.brush_width)
        self.brush_size_scale.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        
        # ---- Prompt Customization Inputs ----
        # Row 5: Building prompt
        subject_label = tk.Label(root, text="Subject:")
        subject_label.grid(row=5, column=0, sticky="w", padx=10)
        self.subject_entry = tk.Entry(root)
        self.subject_entry.insert(0, "")
        self.subject_entry.grid(row=5, column=1, sticky="ew", padx=10)
        
        # Row 6: Design Features prompt
        style_label = tk.Label(root, text="Design Features:")
        style_label.grid(row=6, column=0, sticky="w", padx=10)
        self.style_entry = tk.Entry(root)
        self.style_entry.insert(0, "")
        self.style_entry.grid(row=6, column=1, sticky="ew", padx=10)
        
        # Row 7: Environment prompt
        environment_label = tk.Label(root, text="Environment:")
        environment_label.grid(row=7, column=0, sticky="w", padx=10)
        self.environment_entry = tk.Entry(root)
        self.environment_entry.insert(0, "")
        self.environment_entry.grid(row=7, column=1, sticky="ew", padx=10)
        
        # Row 8: Separate Inpainting Prompt
        inpaint_prompt_label = tk.Label(root, text="Inpainting Prompt:")
        inpaint_prompt_label.grid(row=8, column=0, sticky="w", padx=10)
        self.inpaint_prompt_entry = tk.Entry(root)
        self.inpaint_prompt_entry.insert(0, "")
        self.inpaint_prompt_entry.grid(row=8, column=1, sticky="ew", padx=10)
        
        # Inference is off by default; triggered only on canvas changes.
        self.generated_image = None
        self.is_inference_running = False
        
        # Variables to hold drawing coordinates
        self.last_x, self.last_y = None, None

    def on_button_press(self, event):
        self.last_x, self.last_y = event.x, event.y

    def on_mouse_drag(self, event):
        x, y = event.x, event.y
        if self.eraser_cursor_id is not None:
            # The cursor circle is updated via update_cursor_circle, so no need to delete here
            pass
        # Draw a smooth line on the canvas
        line_id = self.canvas.create_line(
            self.last_x, self.last_y, x, y,
            fill=self.current_color,
            width=self.brush_width,
            capstyle=tk.ROUND,
            smooth=True,
            splinesteps=36
        )
        self.current_stroke_item_ids.append(line_id)
        
        # Also update the PIL drawing
        if self.inpainting_mode:
            if self.inpaint_mask is not None:
                mask_draw = ImageDraw.Draw(self.inpaint_mask)
                mask_draw.line([self.last_x, self.last_y, x, y], fill="white", width=self.brush_width)
                mask_draw.ellipse(
                    (x - self.brush_width//2, y - self.brush_width//2,
                     x + self.brush_width//2, y + self.brush_width//2),
                    fill="white"
                )
        else:
            self.draw.line([self.last_x, self.last_y, x, y],
                           fill=self.current_color, width=self.brush_width)
            # Draw an ellipse at the current point to smooth the stroke
            self.draw.ellipse(
                (x - self.brush_width//2, y - self.brush_width//2,
                 x + self.brush_width//2, y + self.brush_width//2),
                fill=self.current_color
            )
        self.last_x, self.last_y = x, y

    def on_button_release(self, event):
        if not self.inpainting_mode and self.current_stroke_item_ids:
            image_copy = self.sketch.copy()
            self.undo_stack.append({'image': image_copy, 'item_ids': self.current_stroke_item_ids})
        self.current_stroke_item_ids = []
        self.last_x, self.last_y = None, None
        if not self.inpainting_mode:
            self.trigger_inference()

    def on_ctrl_z(self, event):
        if not self.inpainting_mode and self.undo_stack:
            last_state = self.undo_stack.pop()
            self.sketch = last_state['image']
            self.draw = ImageDraw.Draw(self.sketch)
            for item_id in last_state['item_ids']:
                self.canvas.delete(item_id)
            self.trigger_inference()

    def update_cursor_circle(self, event):
        # Always display a preview of the brush size at the cursor
        radius = self.brush_width // 2
        x1 = event.x - radius
        y1 = event.y - radius
        x2 = event.x + radius
        y2 = event.y + radius
        # Use red for eraser, gray for normal drawing
        outline_color = "red" if self.current_color == "white" else "gray"
        if self.eraser_cursor_id is None:
            self.eraser_cursor_id = self.canvas.create_oval(x1, y1, x2, y2, outline=outline_color, dash=(2,2))
        else:
            self.canvas.coords(self.eraser_cursor_id, x1, y1, x2, y2)
            self.canvas.itemconfig(self.eraser_cursor_id, outline=outline_color)

    def update_brush_size(self, value):
        self.brush_width = int(value)

    def get_custom_prompt(self):
        Subject = self.subject_entry.get().strip()
        Design_Features = self.style_entry.get().strip()
        Environment = self.environment_entry.get().strip()
        prompt = (f"a photorealistic render of a [{Subject}], featuring "
                  f"[{Design_Features}] design elements in a [{Environment}] environment, "
                  f"realistic lighting, 8K resolution, cinematic atmosphere, masterpiece.")
        return prompt

    def get_inpaint_prompt(self):
        return self.inpaint_prompt_entry.get().strip()

    def trigger_inference(self):
        if not self.is_inference_running and not self.inpainting_mode:
            self.is_inference_running = True
            thread = threading.Thread(target=self.run_inference)
            thread.start()

    def run_inference(self):
        try:
            custom_prompt = self.get_custom_prompt()
            neg_prompt = ("(drawing, anime, bad photo, bad photography:1.3), "
                          "(worst quality, low quality, blurry:1.2), (bad teeth, deformed teeth, deformed lips), "
                          "(bad anatomy), (deformed iris, deformed pupils), "
                          "(deformed eyes, bad eyes), (deformed face, ugly face, bad face), "
                          "(deformed hands, bad hands, fused fingers), morbid, mutilated, mutation, disfigured")
            
            result_img = generate_preview_image(self.sketch, custom_prompt, neg_prompt)
            self.generated_image = result_img
            self.root.after(0, self.update_output, result_img)
        except Exception as e:
            print("Error during inference:", e)
        finally:
            self.is_inference_running = False

    def trigger_inpaint_process(self):
        if not self.is_inference_running and self.inpainting_mode:
            self.is_inference_running = True
            thread = threading.Thread(target=self.run_inpainting)
            thread.start()

    def run_inpainting(self):
        try:
            custom_inpaint_prompt = self.get_inpaint_prompt()
            neg_prompt = ("(drawing, anime, bad photo, bad photography:1.3), "
                          "(worst quality, low quality, blurry:1.2), (bad teeth, deformed teeth, deformed lips), "
                          "(bad anatomy), (deformed iris, deformed pupils), "
                          "(deformed eyes, bad eyes), (deformed face, ugly face, bad face), "
                          "(deformed hands, bad hands, fused fingers), morbid, mutilated, mutation, disfigured")
            result_img = generate_inpaint_image(self.base_for_inpainting, self.inpaint_mask, custom_inpaint_prompt, neg_prompt)
            self.generated_image = result_img
            self.root.after(0, self.update_output, result_img)
        except Exception as e:
            print("Error during inpainting:", e)
        finally:
            self.is_inference_running = False

    def update_output(self, img):
        self.generated_tk_img = ImageTk.PhotoImage(img)
        self.output_label.config(image=self.generated_tk_img)

    def clear_canvas(self):
        self.canvas.delete("all")
        if self.inpainting_mode:
            # In inpainting mode, clear the mask and redraw the base image.
            if self.base_for_inpainting:
                self.tk_base_img = ImageTk.PhotoImage(self.base_for_inpainting)
                self.canvas.create_image(0, 0, anchor="nw", image=self.tk_base_img)
            if self.inpaint_mask:
                self.inpaint_mask = Image.new("L", (self.canvas_width, self.canvas_height), 0)
        else:
            self.sketch = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
            self.draw = ImageDraw.Draw(self.sketch)
            self.undo_stack.clear()
            self.output_label.config(image="")

    def toggle_eraser(self):
        if self.current_color == "black":
            self.current_color = "white"
            self.eraser_button.config(text="Toggle Draw")
        else:
            self.current_color = "black"
            self.eraser_button.config(text="Toggle Eraser")
            if self.eraser_cursor_id is not None:
                self.canvas.delete(self.eraser_cursor_id)
                self.eraser_cursor_id = None

    def toggle_inpainting_mode(self):
        # Toggle between normal drawing mode and inpainting mode.
        if not self.inpainting_mode:
            # Enter inpainting mode
            if self.generated_image is None:
                messagebox.showwarning("No Base Image", "No generated image available for inpainting!")
                return
            self.inpainting_mode = True
            # Set the base image to the current generated image
            self.base_for_inpainting = self.generated_image.copy()
            # Initialize an empty mask (black background)
            self.inpaint_mask = Image.new("L", (self.canvas_width, self.canvas_height), 0)
            # Clear canvas and display the base image
            self.canvas.delete("all")
            self.tk_base_img = ImageTk.PhotoImage(self.base_for_inpainting)
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_base_img)
            # Change brush color to red for mask drawing
            self.current_color = "red"
            self.inpaint_button.config(state=tk.NORMAL)
            self.toggle_inpaint_mode_button.config(text="Exit Inpainting Mode")
        else:
            # Exit inpainting mode: reset to normal drawing (sketch) mode
            self.inpainting_mode = False
            self.inpaint_mask = None
            self.canvas.delete("all")
            self.current_color = "black"
            self.inpaint_button.config(state=tk.DISABLED)
            self.toggle_inpaint_mode_button.config(text="Enter Inpainting Mode")

    def save_generated(self):
        if self.generated_image is None:
            messagebox.showwarning("No Image", "No generated image to save yet!")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpeg",
            filetypes=[("JPEG files", "*.jpeg"), ("PNG files", "*.png"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self.generated_image.save(file_path)
                messagebox.showinfo("Image Saved", f"Generated image saved as:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save image:\n{e}")

##############################################
# Run the Application
##############################################

if __name__ == "__main__":
    root = tk.Tk()
    app = SketchApp(root)
    root.mainloop()
