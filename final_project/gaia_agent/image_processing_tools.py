import os
import tempfile
import uuid
import base64
from io import BytesIO
from typing import Dict, Any, Optional, List, Tuple
from smolagents import tool

try:
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
    import numpy as np
except ImportError:
    Image = None
    ImageDraw = None
    ImageEnhance = None
    ImageFilter = None
    ImageFont = None
    np = None

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ImportError:
    plt = None
    patches = None

def check_dependencies():
    """Check if required dependencies are available."""
    if Image is None:
        return "Error: PIL (Pillow) library is required for image processing."
    if np is None:
        return "Error: numpy library is required for image processing."
    return None

def decode_image(image_base64: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))

def encode_image(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def save_image(img: Image.Image, subfolder: str = "") -> str:
    """Save PIL Image to temporary file and return path."""
    temp_dir = tempfile.gettempdir()
    if subfolder:
        temp_dir = os.path.join(temp_dir, subfolder)
        os.makedirs(temp_dir, exist_ok=True)
    
    filename = f"image_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(temp_dir, filename)
    img.save(filepath)
    return filepath

@tool
def load_image_from_file(file_path: str) -> Dict[str, Any]:
    """
    Load an image from file and return as base64.
    Args:
        file_path (str): Path to the image file.
    Returns:
        Dict containing base64 encoded image and metadata.
    """
    dep_error = check_dependencies()
    if dep_error:
        return {"error": dep_error}
    
    try:
        if not os.path.exists(file_path):
            return {"error": f"File does not exist: {file_path}"}
        
        img = Image.open(file_path)
        temp_path = save_image(img)
        image_base64 = encode_image(temp_path)
        
        return {
            "image_base64": image_base64,
            "width": img.width,
            "height": img.height,
            "mode": img.mode,
            "format": img.format
        }
    except Exception as e:
        return {"error": f"Error loading image: {str(e)}"}

@tool
def save_image_to_file(image_base64: str, file_path: str) -> Dict[str, Any]:
    """
    Save a base64 encoded image to file.
    Args:
        image_base64 (str): Base64 encoded image string.
        file_path (str): Path where to save the image.
    Returns:
        Dict with save status.
    """
    dep_error = check_dependencies()
    if dep_error:
        return {"error": dep_error}
    
    try:
        img = decode_image(image_base64)
        img.save(file_path)
        return {"success": True, "saved_to": file_path}
    except Exception as e:
        return {"error": f"Error saving image: {str(e)}"}

@tool
def analyze_image(image_base64: str) -> Dict[str, Any]:
    """
    Analyze basic properties of an image (size, mode, color analysis, thumbnail preview).
    Args:
        image_base64 (str): Base64 encoded image string
    Returns:
        Dictionary with analysis result
    """
    dep_error = check_dependencies()
    if dep_error:
        return {"error": dep_error}
    
    try:
        img = decode_image(image_base64)
        width, height = img.size
        mode = img.mode

        analysis = {
            "dimensions": {"width": width, "height": height},
            "mode": mode,
            "aspect_ratio": round(width / height, 2),
            "total_pixels": width * height
        }

        # Color analysis for RGB/RGBA images
        if mode in ("RGB", "RGBA"):
            arr = np.array(img)
            if mode == "RGBA":
                # Handle alpha channel
                rgb_arr = arr[:, :, :3]
                alpha_arr = arr[:, :, 3]
                analysis["has_transparency"] = np.any(alpha_arr < 255)
            else:
                rgb_arr = arr
                analysis["has_transparency"] = False
            
            avg_colors = rgb_arr.mean(axis=(0, 1))
            dominant_channel = ["Red", "Green", "Blue"][np.argmax(avg_colors)]
            brightness = avg_colors.mean()
            
            analysis["color_analysis"] = {
                "average_rgb": [round(c, 1) for c in avg_colors],
                "brightness": round(brightness, 1),
                "dominant_color": dominant_channel,
                "is_dark": brightness < 128,
                "is_bright": brightness > 200
            }
        else:
            analysis["color_analysis"] = {"note": f"No color analysis available for mode {mode}"}

        # Create thumbnail
        thumbnail = img.copy()
        thumbnail.thumbnail((100, 100))
        thumb_path = save_image(thumbnail, "thumbnails")
        thumbnail_base64 = encode_image(thumb_path)
        analysis["thumbnail"] = thumbnail_base64

        return analysis
    except Exception as e:
        return {"error": f"Error analyzing image: {str(e)}"}

@tool
def transform_image(
    image_base64: str, operation: str, params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Apply transformations: resize, rotate, crop, flip, brightness, contrast, blur, sharpen, grayscale.
    Args:
        image_base64 (str): Base64 encoded input image
        operation (str): Transformation operation
        params (Dict[str, Any], optional): Parameters for the operation
    Returns:
        Dictionary with transformed image (base64)
    """
    dep_error = check_dependencies()
    if dep_error:
        return {"error": dep_error}
    
    try:
        img = decode_image(image_base64)
        params = params or {}

        if operation == "resize":
            width = params.get("width", img.width // 2)
            height = params.get("height", img.height // 2)
            resample = getattr(Image, params.get("resample", "LANCZOS"), Image.LANCZOS)
            img = img.resize((width, height), resample)
            
        elif operation == "rotate":
            angle = params.get("angle", 90)
            expand = params.get("expand", True)
            fillcolor = params.get("fillcolor", "white")
            img = img.rotate(angle, expand=expand, fillcolor=fillcolor)
            
        elif operation == "crop":
            left = params.get("left", 0)
            top = params.get("top", 0)
            right = params.get("right", img.width)
            bottom = params.get("bottom", img.height)
            img = img.crop((left, top, right, bottom))
            
        elif operation == "flip":
            direction = params.get("direction", "horizontal")
            if direction == "horizontal":
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            elif direction == "vertical":
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            else:
                return {"error": "Invalid flip direction. Use 'horizontal' or 'vertical'"}
                
        elif operation == "adjust_brightness":
            factor = params.get("factor", 1.5)
            img = ImageEnhance.Brightness(img).enhance(factor)
            
        elif operation == "adjust_contrast":
            factor = params.get("factor", 1.5)
            img = ImageEnhance.Contrast(img).enhance(factor)
            
        elif operation == "adjust_saturation":
            factor = params.get("factor", 1.5)
            img = ImageEnhance.Color(img).enhance(factor)
            
        elif operation == "adjust_sharpness":
            factor = params.get("factor", 1.5)
            img = ImageEnhance.Sharpness(img).enhance(factor)
            
        elif operation == "blur":
            radius = params.get("radius", 2)
            img = img.filter(ImageFilter.GaussianBlur(radius))
            
        elif operation == "sharpen":
            img = img.filter(ImageFilter.SHARPEN)
            
        elif operation == "edge_enhance":
            img = img.filter(ImageFilter.EDGE_ENHANCE)
            
        elif operation == "grayscale":
            img = img.convert("L")
            
        elif operation == "sepia":
            if img.mode != "RGB":
                img = img.convert("RGB")
            arr = np.array(img)
            sepia_filter = np.array([
                [0.393, 0.769, 0.189],
                [0.349, 0.686, 0.168],
                [0.272, 0.534, 0.131]
            ])
            sepia_img = arr.dot(sepia_filter.T)
            sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
            img = Image.fromarray(sepia_img)
            
        else:
            return {"error": f"Unknown operation: {operation}"}

        result_path = save_image(img)
        result_base64 = encode_image(result_path)
        return {
            "transformed_image": result_base64,
            "operation": operation,
            "parameters": params
        }

    except Exception as e:
        return {"error": f"Error transforming image: {str(e)}"}

@tool
def draw_on_image(
    image_base64: str, drawing_type: str, params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Draw shapes (rectangle, circle, line) or text onto an image.
    Args:
        image_base64 (str): Base64 encoded input image
        drawing_type (str): Drawing type (rectangle, circle, line, text, polygon)
        params (Dict[str, Any]): Drawing parameters
    Returns:
        Dictionary with result image (base64)
    """
    dep_error = check_dependencies()
    if dep_error:
        return {"error": dep_error}
    
    try:
        img = decode_image(image_base64)
        draw = ImageDraw.Draw(img)
        color = params.get("color", "red")
        width = params.get("width", 2)

        if drawing_type == "rectangle":
            left = params.get("left", 10)
            top = params.get("top", 10)
            right = params.get("right", 100)
            bottom = params.get("bottom", 100)
            fill = params.get("fill", None)
            draw.rectangle([left, top, right, bottom], outline=color, width=width, fill=fill)
            
        elif drawing_type == "circle":
            x = params.get("x", 50)
            y = params.get("y", 50)
            radius = params.get("radius", 25)
            fill = params.get("fill", None)
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline=color, width=width, fill=fill
            )
            
        elif drawing_type == "line":
            start_x = params.get("start_x", 0)
            start_y = params.get("start_y", 0)
            end_x = params.get("end_x", 100)
            end_y = params.get("end_y", 100)
            draw.line((start_x, start_y, end_x, end_y), fill=color, width=width)
            
        elif drawing_type == "text":
            x = params.get("x", 10)
            y = params.get("y", 10)
            text = params.get("text", "Sample Text")
            font_size = params.get("font_size", 20)
            
            try:
                # Try to load a system font
                font_names = ["arial.ttf", "DejaVuSans.ttf", "liberation-sans.ttf"]
                font = None
                for font_name in font_names:
                    try:
                        font = ImageFont.truetype(font_name, font_size)
                        break
                    except (OSError, IOError):
                        continue
                if font is None:
                    font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
                
            draw.text((x, y), text, fill=color, font=font)
            
        elif drawing_type == "polygon":
            points = params.get("points", [(10, 10), (50, 10), (30, 50)])
            fill = params.get("fill", None)
            draw.polygon(points, outline=color, width=width, fill=fill)
            
        else:
            return {"error": f"Unknown drawing type: {drawing_type}"}

        result_path = save_image(img)
        result_base64 = encode_image(result_path)
        return {
            "result_image": result_base64,
            "drawing_type": drawing_type,
            "parameters": params
        }

    except Exception as e:
        return {"error": f"Error drawing on image: {str(e)}"}

@tool
def generate_simple_image(
    image_type: str,
    width: int = 500,
    height: int = 500,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate a simple image (gradient, noise, pattern, solid_color, checkerboard).
    Args:
        image_type (str): Type of image to generate
        width (int): Image width (default 500)
        height (int): Image height (default 500)
        params (Dict[str, Any], optional): Specific parameters
    Returns:
        Dictionary with generated image (base64)
    """
    dep_error = check_dependencies()
    if dep_error:
        return {"error": dep_error}
    
    try:
        params = params or {}

        if image_type == "solid_color":
            color = params.get("color", (255, 0, 0))  # Default red
            img = Image.new("RGB", (width, height), color)
            
        elif image_type == "gradient":
            direction = params.get("direction", "horizontal")
            start_color = params.get("start_color", (255, 0, 0))
            end_color = params.get("end_color", (0, 0, 255))

            img = Image.new("RGB", (width, height))
            draw = ImageDraw.Draw(img)

            if direction == "horizontal":
                for x in range(width):
                    ratio = x / width
                    r = int(start_color[0] + (end_color[0] - start_color[0]) * ratio)
                    g = int(start_color[1] + (end_color[1] - start_color[1]) * ratio)
                    b = int(start_color[2] + (end_color[2] - start_color[2]) * ratio)
                    draw.line([(x, 0), (x, height)], fill=(r, g, b))
            elif direction == "vertical":
                for y in range(height):
                    ratio = y / height
                    r = int(start_color[0] + (end_color[0] - start_color[0]) * ratio)
                    g = int(start_color[1] + (end_color[1] - start_color[1]) * ratio)
                    b = int(start_color[2] + (end_color[2] - start_color[2]) * ratio)
                    draw.line([(0, y), (width, y)], fill=(r, g, b))
            elif direction == "diagonal":
                for i in range(width + height):
                    ratio = i / (width + height)
                    r = int(start_color[0] + (end_color[0] - start_color[0]) * ratio)
                    g = int(start_color[1] + (end_color[1] - start_color[1]) * ratio)
                    b = int(start_color[2] + (end_color[2] - start_color[2]) * ratio)
                    draw.line([(max(0, i-height), max(0, height-i)), 
                              (min(width, i), min(height, height+width-i))], 
                             fill=(r, g, b))

        elif image_type == "noise":
            noise_type = params.get("noise_type", "random")
            if noise_type == "random":
                noise_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            elif noise_type == "gaussian":
                mean = params.get("mean", 128)
                std = params.get("std", 50)
                noise_array = np.random.normal(mean, std, (height, width, 3))
                noise_array = np.clip(noise_array, 0, 255).astype(np.uint8)
            else:
                noise_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            img = Image.fromarray(noise_array, "RGB")

        elif image_type == "checkerboard":
            square_size = params.get("square_size", 50)
            color1 = params.get("color1", (255, 255, 255))  # White
            color2 = params.get("color2", (0, 0, 0))        # Black
            
            img = Image.new("RGB", (width, height))
            draw = ImageDraw.Draw(img)
            
            for x in range(0, width, square_size):
                for y in range(0, height, square_size):
                    if ((x // square_size) + (y // square_size)) % 2 == 0:
                        color = color1
                    else:
                        color = color2
                    draw.rectangle(
                        [x, y, min(x + square_size, width), min(y + square_size, height)],
                        fill=color
                    )

        else:
            return {"error": f"Unsupported image_type: {image_type}"}

        result_path = save_image(img)
        result_base64 = encode_image(result_path)
        return {
            "generated_image": result_base64,
            "image_type": image_type,
            "dimensions": {"width": width, "height": height},
            "parameters": params
        }

    except Exception as e:
        return {"error": f"Error generating image: {str(e)}"}

@tool
def combine_images(
    images_base64: List[str], operation: str, params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Combine multiple images (stack, grid, blend, overlay).
    Args:
        images_base64 (List[str]): List of base64 images
        operation (str): Combination type
        params (Dict[str, Any], optional): Operation parameters
    Returns:
        Dictionary with combined image (base64)
    """
    dep_error = check_dependencies()
    if dep_error:
        return {"error": dep_error}
    
    try:
        if not images_base64:
            return {"error": "No images provided"}
        
        images = [decode_image(b64) for b64 in images_base64]
        params = params or {}

        if operation == "stack":
            direction = params.get("direction", "horizontal")
            spacing = params.get("spacing", 0)
            background_color = params.get("background_color", "white")
            
            if direction == "horizontal":
                total_width = sum(img.width for img in images) + spacing * (len(images) - 1)
                max_height = max(img.height for img in images)
                new_img = Image.new("RGB", (total_width, max_height), background_color)
                x = 0
                for img in images:
                    # Center vertically
                    y = (max_height - img.height) // 2
                    new_img.paste(img, (x, y))
                    x += img.width + spacing
            else:  # vertical
                max_width = max(img.width for img in images)
                total_height = sum(img.height for img in images) + spacing * (len(images) - 1)
                new_img = Image.new("RGB", (max_width, total_height), background_color)
                y = 0
                for img in images:
                    # Center horizontally
                    x = (max_width - img.width) // 2
                    new_img.paste(img, (x, y))
                    y += img.height + spacing
                    
        elif operation == "grid":
            cols = params.get("cols", 2)
            spacing = params.get("spacing", 10)
            background_color = params.get("background_color", "white")
            
            rows = (len(images) + cols - 1) // cols
            max_width = max(img.width for img in images)
            max_height = max(img.height for img in images)
            
            total_width = cols * max_width + (cols - 1) * spacing
            total_height = rows * max_height + (rows - 1) * spacing
            
            new_img = Image.new("RGB", (total_width, total_height), background_color)
            
            for i, img in enumerate(images):
                row = i // cols
                col = i % cols
                x = col * (max_width + spacing)
                y = row * (max_height + spacing)
                new_img.paste(img, (x, y))
                
        elif operation == "blend":
            if len(images) != 2:
                return {"error": "Blend operation requires exactly 2 images"}
            
            alpha = params.get("alpha", 0.5)
            img1, img2 = images
            
            # Resize to same dimensions
            min_width = min(img1.width, img2.width)
            min_height = min(img1.height, img2.height)
            img1 = img1.resize((min_width, min_height))
            img2 = img2.resize((min_width, min_height))
            
            new_img = Image.blend(img1, img2, alpha)
            
        elif operation == "overlay":
            if len(images) < 2:
                return {"error": "Overlay operation requires at least 2 images"}
            
            base_img = images[0].copy()
            overlay_x = params.get("overlay_x", 0)
            overlay_y = params.get("overlay_y", 0)
            
            for overlay_img in images[1:]:
                # Simple paste (could be enhanced with alpha blending)
                base_img.paste(overlay_img, (overlay_x, overlay_y))
                
            new_img = base_img
            
        else:
            return {"error": f"Unsupported combination operation: {operation}"}

        result_path = save_image(new_img)
        result_base64 = encode_image(result_path)
        return {
            "combined_image": result_base64,
            "operation": operation,
            "num_images": len(images),
            "final_dimensions": {"width": new_img.width, "height": new_img.height}
        }

    except Exception as e:
        return {"error": f"Error combining images: {str(e)}"}

@tool
def get_image_histogram(image_base64: str, channel: str = "all") -> Dict[str, Any]:
    """
    Generate histogram data for image color channels.
    Args:
        image_base64 (str): Base64 encoded image
        channel (str): Channel to analyze ("red", "green", "blue", "gray", "all")
    Returns:
        Dictionary with histogram data
    """
    dep_error = check_dependencies()
    if dep_error:
        return {"error": dep_error}
    
    try:
        img = decode_image(image_base64)
        
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
            
        histograms = {}
        
        if channel == "all" or channel == "gray":
            gray_img = img.convert("L")
            hist = gray_img.histogram()
            histograms["gray"] = hist
            
        if img.mode in ("RGB", "RGBA") and channel != "gray":
            if channel in ("all", "red"):
                r_hist = img.getchannel("R").histogram()
                histograms["red"] = r_hist
                
            if channel in ("all", "green"):
                g_hist = img.getchannel("G").histogram()
                histograms["green"] = g_hist
                
            if channel in ("all", "blue"):
                b_hist = img.getchannel("B").histogram()
                histograms["blue"] = b_hist
        
        # Calculate basic statistics
        stats = {}
        for ch, hist in histograms.items():
            total_pixels = sum(hist)
            weighted_sum = sum(i * count for i, count in enumerate(hist))
            mean_val = weighted_sum / total_pixels if total_pixels > 0 else 0
            stats[ch] = {
                "mean": round(mean_val, 2),
                "min_value": next((i for i, count in enumerate(hist) if count > 0), 0),
                "max_value": next((255 - i for i, count in enumerate(reversed(hist)) if count > 0), 255)
            }
        
        return {
            "histograms": histograms,
            "statistics": stats,
            "image_mode": img.mode,
            "total_pixels": img.width * img.height
        }
        
    except Exception as e:
        return {"error": f"Error generating histogram: {str(e)}"}