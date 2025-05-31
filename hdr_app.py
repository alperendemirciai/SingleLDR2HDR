import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import io
import os
from pathlib import Path
import cv2

# Import your models (adjust paths as needed)
try:
    import models.model_a
    import models.model_b
    import models.unet
except ImportError as e:
    st.error(f"Error importing models: {e}")
    st.stop()


def load_checkpoint(filepath, model, optimizer=None, lr=None, device=None):
    """
    Load model checkpoint
    
    Args:
        filepath: Path to the checkpoint file
        model: PyTorch model to load parameters into
        optimizer: PyTorch optimizer to load state into (optional)
        lr: Learning rate to set (optional)
        device: Device to load the model on (optional)
        
    Returns:
        epoch: The epoch number of the loaded checkpoint
    """
    if not os.path.exists(filepath):
        print(f"Checkpoint file not found at {filepath}")
        return 0
        
    if device is None:
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    return checkpoint['epoch']


def load_model(model_type, checkpoint_path, upsampling_method='bilinear', device='cpu'):
    """Load the specified model with checkpoint"""
    try:
        if model_type == 'unet':
            model = models.unet.UNet(
                in_channels=3, 
                out_channels=3, 
                base_filters=16, 
                upsampling_method=upsampling_method
            ).to(device)
        elif model_type == 'model_a':
            model = models.model_a.Autoencoder(
                upsample_mode=upsampling_method
            ).to(device)
        elif model_type == 'model_b':
            model = models.model_b.MobileNetV3_UNet(3).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint
        if os.path.exists(checkpoint_path):
            load_checkpoint(checkpoint_path, model, device=device)
            return model
        else:
            st.error(f"Checkpoint not found at {checkpoint_path}")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def crop_center_patch(image, patch_size=512):
    """Crop a center patch from the image"""
    width, height = image.size
    
    # Calculate crop coordinates for center patch
    left = max(0, (width - patch_size) // 2)
    top = max(0, (height - patch_size) // 2)
    right = min(width, left + patch_size)
    bottom = min(height, top + patch_size)
    
    # Crop the center patch
    cropped = image.crop((left, top, right, bottom))
    
    # If the cropped image is smaller than patch_size, pad with black
    if cropped.size[0] < patch_size or cropped.size[1] < patch_size:
        padded = Image.new('RGB', (patch_size, patch_size), (0, 0, 0))
        paste_x = (patch_size - cropped.size[0]) // 2
        paste_y = (patch_size - cropped.size[1]) // 2
        padded.paste(cropped, (paste_x, paste_y))
        return padded
    
    return cropped


def resize_with_padding(image, target_size=512):
    """Resize image maintaining aspect ratio and pad to square with black pixels"""
    width, height = image.size
    aspect_ratio = width / height
    
    # Calculate new dimensions keeping longest edge as target_size
    if width > height:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_height = target_size
        new_width = int(target_size * aspect_ratio)
    
    # Resize the image
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create a black square canvas
    padded = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    
    # Calculate position to paste the resized image (center it)
    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2
    
    # Paste the resized image onto the black canvas
    padded.paste(resized, (paste_x, paste_y))
    
    return padded


def preprocess_image(image, target_size=(512, 512), processing_mode='resize'):
    """Preprocess uploaded image for model input"""
    if processing_mode == 'crop':
        # Crop center patch
        processed_image = crop_center_patch(image, target_size[0])
    elif processing_mode == 'resize_pad':
        # Resize with padding
        processed_image = resize_with_padding(image, target_size[0])
    else:
        # Default: simple resize (stretching)
        processed_image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    tensor = transform(processed_image).unsqueeze(0)  # Add batch dimension
    return tensor * 2.0 - 1.0, processed_image  # Return both tensor and processed PIL image


def apply_opencv_tonemap(hdr_array, method='reinhard', **kwargs):
    """Apply OpenCV tone mapping functions"""
    # Ensure input is float32 and in valid range
    hdr_float = hdr_array.astype(np.float32)
    hdr_float = np.clip(hdr_float, 0, None)  # Remove negative values
    
    if method == 'reinhard':
        # OpenCV Reinhard tone mapping
        gamma = kwargs.get('gamma', 1.0)
        intensity = kwargs.get('intensity', 0.0)
        light_adapt = kwargs.get('light_adapt', 1.0)
        color_adapt = kwargs.get('color_adapt', 0.0)
        
        tonemap = cv2.createTonemapReinhard(
            gamma=gamma,
            intensity=intensity,
            light_adapt=light_adapt,
            color_adapt=color_adapt
        )
        result = tonemap.process(hdr_float)
        
    elif method == 'drago':
        # OpenCV Drago tone mapping
        gamma = kwargs.get('gamma', 1.0)
        saturation = kwargs.get('saturation', 1.0)
        bias = kwargs.get('bias', 0.85)
        
        tonemap = cv2.createTonemapDrago(
            gamma=gamma,
            saturation=saturation,
            bias=bias
        )
        result = tonemap.process(hdr_float)
        
    elif method == 'mantiuk':
        # OpenCV Mantiuk tone mapping
        gamma = kwargs.get('gamma', 1.0)
        scale = kwargs.get('scale', 0.7)
        saturation = kwargs.get('saturation', 1.0)
        
        tonemap = cv2.createTonemapMantiuk(
            gamma=gamma,
            scale=scale,
            saturation=saturation
        )
        result = tonemap.process(hdr_float)
        
    elif method == 'linear':
        # Simple linear mapping
        result = np.clip(hdr_float, 0, 1)
        
    elif method == 'gamma':
        # Gamma correction
        gamma = kwargs.get('gamma', 2.2)
        result = np.power(np.clip(hdr_float, 0, 1), 1.0/gamma)
        
    else:
        # Default to linear
        result = np.clip(hdr_float, 0, 1)
    
    # Ensure output is in [0, 1] range
    result = np.clip(result, 0, 1)
    return result


def postprocess_output(output_tensor, tone_mapping='reinhard', apply_bgr_flip=False, **tone_params):
    """Convert model output back to displayable image with OpenCV tone mapping"""
    # Remove batch dimension and move to CPU
    output = output_tensor.squeeze(0).cpu()
    
    # Apply BGR/RGB flip using torch.flip if needed (following your example)
    if apply_bgr_flip:
        output = torch.flip(output, dims=[0])  # Flip channel dimension
    
    # Denormalize from [-1, 1] to [0, 1]
    output = (output + 1.0) / 2.0
    
    # Convert to numpy array (C, H, W) -> (H, W, C)
    output_np = output.permute(1, 2, 0).numpy()
    
    # For HDR processing, we might need to recover from log space
    # Uncomment if your model outputs log-space HDR
    # output_np = np.exp(output_np) + 1.0
    
    # Apply OpenCV tone mapping
    tone_mapped = apply_opencv_tonemap(output_np, method=tone_mapping, **tone_params)
    
    # Convert to PIL Image (expects RGB)
    output_image = Image.fromarray((tone_mapped * 255).astype(np.uint8))
    
    return output_image


def create_comparison_plot(original, processed, tone_mapping_method):
    """Create side-by-side comparison plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(original)
    ax1.set_title('Original Image (512x512)')
    ax1.axis('off')
    
    ax2.imshow(processed)
    ax2.set_title(f'HDR Processed ({tone_mapping_method.title()} Tone Mapping)')
    ax2.axis('off')
    
    plt.tight_layout()
    return fig


def main():
    st.set_page_config(
        page_title="HDR Image Processing",
        page_icon="🌅",
        layout="wide"
    )
    
    st.title("🌅 HDR Image Processing App")
    st.markdown("Upload an image and process it with your HDR model using OpenCV tone mapping")
    
    # Sidebar for model configuration
    with st.sidebar:
        st.header("Image Processing Options")
        
        # Image processing mode selection
        processing_mode = st.selectbox(
            "Image Processing Mode",
            options=['resize', 'crop', 'resize_pad'],
            index=0,
            help="Choose how to prepare the image for 512x512 processing"
        )
        
        if processing_mode == 'resize':
            st.info("🔄 **Resize Mode**: Image will be stretched/squeezed to 512x512 (may distort aspect ratio)")
        elif processing_mode == 'crop':
            st.info("✂️ **Crop Mode**: Center 512x512 patch will be cropped from the image")
        elif processing_mode == 'resize_pad':
            st.info("📐 **Resize + Pad Mode**: Longest edge resized to 512px, then padded with black pixels to 512x512")
        
        st.header("Model Configuration")
        
        # Model selection
        model_type = st.selectbox(
            "Select Model",
            options=['unet', 'model_a', 'model_b'],
            index=0
        )
        
        # Upsampling method
        upsampling_method = st.selectbox(
            "Upsampling Method",
            options=['bilinear', 'nearest', 'bicubic'],
            index=0
        )
        
        # Checkpoint path
        checkpoint_path = st.text_input(
            "Checkpoint Path",
            value="outputs_01hdr/9th_run_model_b/checkpoints/gen_best.pth",
            help="Path to your model checkpoint file"
        )
        
        # Device selection
        device_options = ['cpu']
        if torch.cuda.is_available():
            device_options.append('cuda')
        if torch.backends.mps.is_available():
            device_options.append('mps')
        
        device = st.selectbox("Device", options=device_options)
        
        st.header("Tone Mapping Options")
        
        # Tone mapping method selection (OpenCV methods)
        tone_mapping_method = st.selectbox(
            "OpenCV Tone Mapping Method",
            options=['reinhard', 'drago', 'mantiuk', 'linear', 'gamma'],
            index=0,
            help="Choose OpenCV tone mapping technique for HDR visualization"
        )
        
        # Method-specific parameters
        tone_params = {}
        
        if tone_mapping_method == 'reinhard':
            tone_params['gamma'] = st.slider("Gamma", 0.1, 3.0, 1.0, 0.1)
            tone_params['intensity'] = st.slider("Intensity", -8.0, 8.0, 0.0, 0.1)
            tone_params['light_adapt'] = st.slider("Light Adaptation", 0.0, 1.0, 1.0, 0.05)
            tone_params['color_adapt'] = st.slider("Color Adaptation", 0.0, 1.0, 0.0, 0.05)
            
        elif tone_mapping_method == 'drago':
            tone_params['gamma'] = st.slider("Gamma", 0.1, 3.0, 1.0, 0.1)
            tone_params['saturation'] = st.slider("Saturation", 0.0, 2.0, 1.0, 0.05)
            tone_params['bias'] = st.slider("Bias", 0.1, 1.0, 0.85, 0.05)
            
        elif tone_mapping_method == 'mantiuk':
            tone_params['gamma'] = st.slider("Gamma", 0.1, 3.0, 1.0, 0.1)
            tone_params['scale'] = st.slider("Scale", 0.1, 1.0, 0.7, 0.05)
            tone_params['saturation'] = st.slider("Saturation", 0.0, 2.0, 1.0, 0.05)
            
        elif tone_mapping_method == 'gamma':
            tone_params['gamma'] = st.slider("Gamma", 0.1, 5.0, 2.2, 0.1)
        
        # Debug option
        show_debug = st.checkbox("Show Debug Info", help="Display tensor and color channel information")
        
        # Color channel flip option (using torch.flip like in your code)
        apply_bgr_flip = st.checkbox("Apply BGR/RGB Flip", 
                                   help="Use torch.flip to swap color channels (following your save_some_examples pattern)")
        
        # HDR recovery option
        is_log_space = st.checkbox("Model outputs log-space HDR", 
                                 help="Check if your model outputs log-space HDR that needs exp() recovery")
    
    # Main content area
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image to process with the HDR model"
    )
    
    if uploaded_file is not None:
        # Display original image info
        image = Image.open(uploaded_file).convert('RGB')
        original_size = image.size
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, caption=f"Size: {original_size[0]} x {original_size[1]}")
            
            # Show preview of how the image will be processed
            if processing_mode == 'crop':
                preview = crop_center_patch(image, 512)
                st.image(preview, caption="Preview: Center 512x512 crop", width=200)
            elif processing_mode == 'resize_pad':
                preview = resize_with_padding(image, 512)
                st.image(preview, caption="Preview: Resized with black padding", width=200)
        
        # Process button
        if st.button("🚀 Process Image", type="primary"):
            with st.spinner("Loading model and processing image..."):
                try:
                    # Load model
                    model = load_model(model_type, checkpoint_path, upsampling_method, device)
                    
                    if model is not None:
                        # Preprocess image with selected mode
                        input_tensor, processed_pil = preprocess_image(
                            image, target_size=(512, 512), processing_mode=processing_mode
                        )
                        input_tensor = input_tensor.to(device)
                        
                        # Show processed input image
                        st.subheader("Processed Input")
                        st.image(processed_pil, caption=f"Input to model: 512x512 ({processing_mode} mode)", width=300)
                        
                        # Run inference
                        model.eval()
                        with torch.no_grad():
                            output_tensor = model(input_tensor)
                        
                        # Debug information
                        if show_debug:
                            st.subheader("🔍 Debug Information")
                            debug_col1, debug_col2 = st.columns(2)
                            
                            with debug_col1:
                                st.write("**Input Tensor Info:**")
                                st.write(f"Shape: {input_tensor.shape}")
                                st.write(f"Min/Max: {input_tensor.min():.3f} / {input_tensor.max():.3f}")
                                st.write(f"Device: {input_tensor.device}")
                            
                            with debug_col2:
                                st.write("**Output Tensor Info:**")
                                st.write(f"Shape: {output_tensor.shape}")
                                st.write(f"Min/Max: {output_tensor.min():.3f} / {output_tensor.max():.3f}")
                                st.write(f"BGR Flip Applied: {apply_bgr_flip}")
                                st.write(f"Log Space Recovery: {is_log_space}")
                        
                        # Create a modified postprocess function for log space
                        def postprocess_with_log_recovery(tensor, tone_mapping, apply_flip, **params):
                            output = tensor.squeeze(0).cpu()
                            
                            if apply_flip:
                                output = torch.flip(output, dims=[0])
                            
                            output = (output + 1.0) / 2.0
                            output_np = output.permute(1, 2, 0).numpy()
                            
                            # HDR recovery from log space if needed
                            if is_log_space:
                                output_np = np.exp(output_np) + 1.0
                            
                            tone_mapped = apply_opencv_tonemap(output_np, method=tone_mapping, **params)
                            return Image.fromarray((tone_mapped * 255).astype(np.uint8))
                        
                        # Postprocess output with OpenCV tone mapping
                        processed_image = postprocess_with_log_recovery(
                            output_tensor, tone_mapping_method, apply_bgr_flip, **tone_params
                        )
                        
                        # Display results
                        with col2:
                            st.subheader("HDR Processed")
                            st.image(processed_image, caption="Size: 512 x 512")
                        
                        # Create comparison plot
                        st.subheader("Comparison")
                        comparison_fig = create_comparison_plot(processed_pil, processed_image, tone_mapping_method)
                        st.pyplot(comparison_fig)
                        
                        # Show multiple OpenCV tone mapping previews
                        st.subheader("OpenCV Tone Mapping Previews")
                        preview_methods = ['reinhard', 'drago', 'mantiuk', 'linear', 'gamma']
                        
                        preview_cols = st.columns(len(preview_methods))
                        for i, method in enumerate(preview_methods):
                            with preview_cols[i]:
                                if method == 'reinhard':
                                    preview_params = {'gamma': 1.0, 'intensity': 0.0, 'light_adapt': 1.0, 'color_adapt': 0.0}
                                elif method == 'drago':
                                    preview_params = {'gamma': 1.0, 'saturation': 1.0, 'bias': 0.85}
                                elif method == 'mantiuk':
                                    preview_params = {'gamma': 1.0, 'scale': 0.7, 'saturation': 1.0}
                                elif method == 'gamma':
                                    preview_params = {'gamma': 2.2}
                                else:
                                    preview_params = {}
                                
                                preview_image = postprocess_with_log_recovery(
                                    output_tensor, method, apply_bgr_flip, **preview_params
                                )
                                st.image(preview_image, caption=method.title(), use_column_width=True)
                        
                        # Download button for processed image
                        buf = io.BytesIO()
                        processed_image.save(buf, format='PNG')
                        buf.seek(0)
                        
                        st.download_button(
                            label="📥 Download Processed Image",
                            data=buf.getvalue(),
                            file_name=f"hdr_{tone_mapping_method}_{uploaded_file.name}",
                            mime="image/png"
                        )
                        
                        st.success("✅ Image processed successfully using OpenCV tone mapping!")
                    
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
    
    # Instructions
    with st.expander("📋 Instructions"):
        st.markdown("""
        1. **Choose Processing Mode**: Select how to prepare your image for 512x512 processing:
           - **Resize**: Stretch/squeeze to 512x512 (may distort aspect ratio)
           - **Crop**: Extract center 512x512 patch (may lose edge content)
           - **Resize + Pad**: Maintain aspect ratio, resize longest edge to 512px, pad with black
        2. **Configure Model**: Select your model type and set the checkpoint path in the sidebar
        2. **Choose OpenCV Tone Mapping**: Select tone mapping method and adjust OpenCV-specific parameters
        3. **Color Channel Options**: Enable BGR/RGB flip using torch.flip() if needed
        4. **Upload Image**: Choose an image file (PNG, JPG, etc.)
        5. **Process**: Click the "Process Image" button to run your HDR model
        6. **Compare**: View different OpenCV tone mapping results in the preview section
        7. **Download**: Save the processed image with your chosen tone mapping
        
        **Note**: All processing modes result in 512x512 pixel input for the model.
        - **Resize**: Fast but may distort proportions
        - **Crop**: Preserves quality but may lose important edge content  
        - **Resize + Pad**: Best quality preservation, maintains aspect ratio
        This version uses OpenCV's built-in tone mapping functions and torch.flip() for color channel handling.
        """)
    
    # OpenCV tone mapping details
    with st.expander("🎨 OpenCV Tone Mapping Methods"):
        st.markdown("""
        - **Reinhard**: cv2.createTonemapReinhard() - Classic tone mapping with gamma, intensity, and adaptation controls
        - **Drago**: cv2.createTonemapDrago() - Adaptive logarithmic mapping with bias control
        - **Mantiuk**: cv2.createTonemapMantiuk() - Perceptual tone mapping with scale and saturation
        - **Linear**: Simple clipping to [0,1] range (non-OpenCV fallback)
        - **Gamma**: Standard gamma correction (non-OpenCV fallback)
        
        **OpenCV Functions Used:**
        - `cv2.createTonemapReinhard(gamma, intensity, light_adapt, color_adapt)`
        - `cv2.createTonemapDrago(gamma, saturation, bias)`
        - `cv2.createTonemapMantiuk(gamma, scale, saturation)`
        - `torch.flip(tensor, dims=[0])` for BGR/RGB channel flipping
        """)
    
    # Technical details
    with st.expander("🔧 Technical Details"):
        st.markdown(f"""
        - **Processing Mode**: {processing_mode if 'processing_mode' in locals() else 'Not selected'}
        - **Model Type**: {model_type if 'model_type' in locals() else 'Not selected'}
        - **Upsampling Method**: {upsampling_method if 'upsampling_method' in locals() else 'Not selected'}
        - **Device**: {device if 'device' in locals() else 'Not selected'}
        - **OpenCV Tone Mapping**: {tone_mapping_method if 'tone_mapping_method' in locals() else 'Not selected'}
        - **BGR/RGB Flip**: Using torch.flip() following your save_some_examples pattern
        - **HDR Recovery**: Optional exp() + 1.0 for log-space models
        - **Input Size**: 512 x 512 pixels
        - **Normalization**: ImageNet normalization + scaling to [-1, 1]
        """)


if __name__ == "__main__":
    main()