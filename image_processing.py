from sklearn.cluster import KMeans
import numpy as np
from PIL import Image, ExifTags
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Constants
OUTPUT_WIDTH = 160
OUTPUT_HEIGHT = 144
BLOCK_WIDTH = 8
BLOCKS_HORIZONTAL = OUTPUT_WIDTH // BLOCK_WIDTH
BLOCKS_VERTICAL = OUTPUT_HEIGHT // BLOCK_WIDTH
NUM_BLOCKS = BLOCKS_HORIZONTAL * BLOCKS_VERTICAL


def resize_and_pad(image, output_width, output_height):
    """Resize and pad image to fit the desired output dimensions, preserving orientation."""
    # Correct orientation based on EXIF metadata
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif and orientation in exif:
            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # If EXIF data is missing or invalid, do nothing
        pass

    # Proceed with resizing and padding
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height

    # Resize while maintaining aspect ratio
    if output_width / output_height > aspect_ratio:
        new_height = output_height
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = output_width
        new_height = int(new_width / aspect_ratio)

    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create a black canvas and paste the resized image
    padded_image = Image.new("RGB", (output_width, output_height), (0, 0, 0))
    offset_x = (output_width - new_width) // 2
    offset_y = (output_height - new_height) // 2
    padded_image.paste(image, (offset_x, offset_y))

    return np.array(padded_image) / 255.0  # Normalize to [0, 1]


def extract_blocks(image):
    """Extract 8x8 blocks from the image."""
    blocks = []
    for i in range(BLOCKS_VERTICAL):
        for j in range(BLOCKS_HORIZONTAL):
            x0 = i * BLOCK_WIDTH
            y0 = j * BLOCK_WIDTH
            block = image[x0:x0 + BLOCK_WIDTH, y0:y0 + BLOCK_WIDTH]
            blocks.append(block)
    return np.array(blocks)  # Shape: (NUM_BLOCKS, BLOCK_WIDTH, BLOCK_WIDTH, 3)


def cluster_blocks(blocks, palette, num_unique_blocks=192):
    """Cluster blocks into 192 unique blocks using k-means with palette constraints."""
    flattened_blocks = blocks.reshape(NUM_BLOCKS, -1)  # Shape: (NUM_BLOCKS, BLOCK_WIDTH * BLOCK_WIDTH * 3)

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_unique_blocks, random_state=42).fit(flattened_blocks)

    # Map each cluster center to the nearest palette color
    def map_to_palette(center):
        reshaped_center = center.reshape(-1, 3)
        mapped_colors = np.array([
            palette[np.argmin(np.linalg.norm(pixel - palette, axis=1))] for pixel in reshaped_center
        ])
        return mapped_colors.flatten()

    unique_blocks = np.array([map_to_palette(center) for center in kmeans.cluster_centers_])

    # Assign blocks to nearest adjusted centers
    labels = kmeans.labels_
    clustered_blocks = unique_blocks[labels].reshape(NUM_BLOCKS, BLOCK_WIDTH, BLOCK_WIDTH, 3)

    return clustered_blocks, unique_blocks.reshape(num_unique_blocks, BLOCK_WIDTH, BLOCK_WIDTH, 3)




def reconstruct_image_from_blocks(blocks):
    """Reconstruct the image from 8x8 blocks."""
    img_height = BLOCKS_VERTICAL * BLOCK_WIDTH
    img_width = BLOCKS_HORIZONTAL * BLOCK_WIDTH
    reconstructed = np.zeros((img_height, img_width, 3))
    for idx, block in enumerate(blocks):
        i = idx // BLOCKS_HORIZONTAL
        j = idx % BLOCKS_HORIZONTAL
        x0 = i * BLOCK_WIDTH
        y0 = j * BLOCK_WIDTH
        reconstructed[x0:x0 + BLOCK_WIDTH, y0:y0 + BLOCK_WIDTH] = block
    return reconstructed


def process_image(image_path, custom_palette=None):
    """Process the image and apply Gameboy-style clustering."""
    image = Image.open(image_path)
    img = resize_and_pad(image, OUTPUT_WIDTH, OUTPUT_HEIGHT)

    # Step 1: Global 4-Color Clustering
    pixels = img.reshape(-1, 3)  # Flatten image
    kmeans = KMeans(n_clusters=4, random_state=42).fit(pixels)
    palette = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Replace colors if custom palette is provided
    if custom_palette:
        palette = np.array([hex_to_rgb(hex_color) for hex_color in custom_palette])
        if len(palette) == 0:
            raise ValueError("Provided custom_palette is empty or invalid.")

    # Replace clustered image colors with the final palette
    clustered_img = palette[labels].reshape(img.shape)

    # Step 3: Extract Blocks
    blocks = extract_blocks(clustered_img)

    # Step 4: Cluster Blocks into 192 Unique Blocks
    clustered_blocks, unique_blocks = cluster_blocks(blocks, palette)  # Pass palette instead of custom_palette

    # Step 5: Reconstruct Image
    reconstructed_img = reconstruct_image_from_blocks(clustered_blocks)

    return reconstructed_img, palette, unique_blocks


def blocks_save_image(array, output_path, save_as_grid=False, grid_cols=16, spacing=2):
    """Save a NumPy array as an image. Optionally, arrange in a grid for unique blocks with spacing."""
    if save_as_grid:
        # Use high-quality visualization for unique blocks
        save_blocks_high_quality(array, output_path, grid_cols, spacing)
    else:
        img = Image.fromarray((array * 255).astype(np.uint8))
        img.save(output_path)


def save_blocks_high_quality(blocks, output_path, grid_cols, spacing):
    """Save unique blocks in a grid format with high-quality rendering, including spacing on all sides."""
    num_blocks, block_width, block_height, _ = blocks.shape
    grid_rows = (num_blocks + grid_cols - 1) // grid_cols  # Calculate required grid rows

    # Calculate the padded width and height for blocks, including spacing
    padded_width = block_width + spacing
    padded_height = block_height + spacing

    # Adjust grid dimensions to include spacing on all sides
    grid = np.ones((
        grid_rows * padded_height + spacing,  # Add extra spacing to the bottom
        grid_cols * padded_width + spacing,   # Add extra spacing to the right
        3
    )) * 1.0  # White background for spacing

    for idx, block in enumerate(blocks):
        row = idx // grid_cols
        col = idx % grid_cols
        start_row = row * padded_height + spacing  # Offset for top spacing
        start_col = col * padded_width + spacing   # Offset for left spacing
        # Correctly place the block into the grid
        grid[start_row:start_row + block_height, start_col:start_col + block_width] = block

    # Save the grid as a high-quality image
    fig, ax = plt.subplots(figsize=(grid_cols, grid_rows), dpi=300)
    ax.axis('off')  # No axes
    ax.imshow(grid, interpolation='nearest')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)



def hex_to_rgb(hex_color):
    """Convert a hex color string to normalized RGB."""
    hex_color = hex_color.lstrip('#')
    return [int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


def save_image(array, output_path):
    """Save a NumPy array as an image."""
    img = Image.fromarray((array * 255).astype(np.uint8))
    img.save(output_path)
