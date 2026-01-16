use crate::config::Config;
use crate::error::Result;
use image::{Rgba, RgbaImage};
use kmeans_colors::get_kmeans;
use palette::{white_point::D65, FromColor, Lab, Srgba};

/// Quantizes the image to a fixed number of colors using K-means clustering.
/// Uses Lab color space for better perceptual color selection.
pub fn quantize_image(img: &RgbaImage, config: &Config) -> Result<RgbaImage> {
    if config.k_colors == 0 {
        return Err(crate::error::PixelSnapperError::InvalidInput(
            "Number of colors must be greater than 0".to_string(),
        ));
    }

    let width = img.width();
    let height = img.height();
    let pixels: Vec<[u8; 4]> = img.pixels().map(|p| p.0).collect();

    let opaque_indices: Vec<usize> = pixels
        .iter()
        .enumerate()
        .filter_map(|(i, p)| if p[3] > 0 { Some(i) } else { None })
        .collect();

    if opaque_indices.is_empty() {
        return Ok(img.clone());
    }

    // Convert opaque pixels to Lab for better perceptual clustering
    let lab_pixels: Vec<Lab<D65, f32>> = opaque_indices
        .iter()
        .map(|&i| {
            let p = pixels[i];
            let srgba = Srgba::new(
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
                p[3] as f32 / 255.0,
            );
            Lab::from_color(srgba.into_linear::<f32, f32>())
        })
        .collect();

    let k = config.k_colors.min(lab_pixels.len());
    let max_iter = config.max_kmeans_iterations;
    let converge = 0.01;
    let verbose = false;
    let seed = config.k_seed;

    // Perform K-means clustering in Lab space
    let result = get_kmeans(k, max_iter, converge, verbose, &lab_pixels, seed);

    // Map pixels back to their closest centroid
    let mut quantized_pixels = pixels.clone();

    for (idx, &pixel_idx) in opaque_indices.iter().enumerate() {
        let centroid_idx = result.indices[idx];
        let lab_centroid = result.centroids[centroid_idx as usize];
        let srgba_centroid: Srgba = Srgba::from_color(lab_centroid);

        // Preserve original alpha
        let original_alpha = pixels[pixel_idx][3];
        quantized_pixels[pixel_idx] = [
            (srgba_centroid.red * 255.0).round() as u8,
            (srgba_centroid.green * 255.0).round() as u8,
            (srgba_centroid.blue * 255.0).round() as u8,
            original_alpha,
        ];
    }

    let mut new_img = RgbaImage::new(width, height);
    for (i, p) in quantized_pixels.iter().enumerate() {
        let x = (i as u32) % width;
        let y = (i as u32) / width;
        new_img.put_pixel(x, y, Rgba(*p));
    }

    Ok(new_img)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn test_quantize_image_zero_colors() {
        let img = RgbaImage::new(10, 10);
        let config = Config {
            k_colors: 0,
            ..Config::default()
        };
        let result = quantize_image(&img, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantize_image_empty_image() {
        let img = RgbaImage::new(10, 10); // Fully transparent
        let config = Config::default();
        let result = quantize_image(&img, &config).unwrap();
        assert_eq!(result.dimensions(), (10, 10));
    }

    #[test]
    fn test_quantize_image_single_color() {
        let mut img = RgbaImage::new(10, 10);
        for p in img.pixels_mut() {
            *p = Rgba([255, 0, 0, 255]);
        }
        let config = Config {
            k_colors: 1,
            ..Config::default()
        };
        let result = quantize_image(&img, &config).unwrap();
        // Lab conversion/quantization should be very close to original red
        let p = result.get_pixel(0, 0);
        assert!(p[0] > 250);
        assert!(p[1] < 10);
        assert!(p[2] < 10);
        assert_eq!(p[3], 255);
    }
}
