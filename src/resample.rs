use crate::error::{PixelSnapperError, Result};
use crate::quantize::MAX_PALETTE;
use image::{ImageBuffer, RgbaImage};
use rayon::prelude::*;

/// Resample using pre-built palette and indexed buffer from quantization.
/// This eliminates the O(N) palette-building scan that was previously done here.
pub fn resample(
    img: &RgbaImage,
    cols: &[usize],
    rows: &[usize],
    palette: &[[u8; 4]],
    indexed: &[u8],
) -> Result<RgbaImage> {
    if cols.len() < 2 || rows.len() < 2 {
        return Err(PixelSnapperError::ProcessingError(
            "Insufficient grid cuts for resampling".to_string(),
        ));
    }

    let out_w = (cols.len().max(1) - 1) as u32;
    let out_h = (rows.len().max(1) - 1) as u32;

    let in_width = img.width() as usize;
    let in_height = img.height() as usize;
    let palette_size = palette.len();

    let mut final_img: RgbaImage = ImageBuffer::new(out_w, out_h);

    {
        let w = out_w;
        let samples = final_img.as_flat_samples_mut().samples;

        samples
            .par_chunks_exact_mut(4)
            .enumerate()
            .for_each(|(idx, pixel_sample)| {
                let x_i = (idx as u32 % w) as usize;
                let y_i = (idx as u32 / w) as usize;

                let ys = rows[y_i];
                let ye = rows[y_i + 1];
                let xs = cols[x_i];
                let xe = cols[x_i + 1];

                let best_pixel = if xe <= xs || ye <= ys {
                    [0, 0, 0, 0]
                } else if xe - xs == 1 && ye - ys == 1 {
                    // 1:1 fast path
                    if xs < in_width && ys < in_height {
                        let idx = indexed[ys * in_width + xs];
                        if (idx as usize) < palette_size {
                            palette[idx as usize]
                        } else {
                            [0, 0, 0, 0]
                        }
                    } else {
                        [0, 0, 0, 0]
                    }
                } else {
                    // Zero-allocation fixed array counting
                    let mut counts = [0u32; MAX_PALETTE];

                    for y in ys..ye.min(in_height) {
                        let row_offset = y * in_width;
                        for x in xs..xe.min(in_width) {
                            let color_idx = indexed[row_offset + x];
                            if (color_idx as usize) < palette_size {
                                counts[color_idx as usize] += 1;
                            }
                        }
                    }

                    // Find mode in O(K)
                    let mut best_idx = 0;
                    let mut max_count = 0u32;
                    for (i, &count) in counts[..palette_size].iter().enumerate() {
                        if count > max_count {
                            max_count = count;
                            best_idx = i;
                        }
                    }

                    if max_count > 0 {
                        palette[best_idx]
                    } else {
                        [0, 0, 0, 0]
                    }
                };

                pixel_sample[0] = best_pixel[0];
                pixel_sample[1] = best_pixel[1];
                pixel_sample[2] = best_pixel[2];
                pixel_sample[3] = best_pixel[3];
            });
    }

    Ok(final_img)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_insufficient_grid() {
        let img = RgbaImage::new(10, 10);
        let palette = vec![[255, 0, 0, 255]];
        let indexed = vec![0u8; 100];
        let result = resample(&img, &vec![0], &vec![0, 10], &palette, &indexed);
        assert!(result.is_err());
    }

    #[test]
    fn test_resample_simple_grid() {
        let mut img = RgbaImage::new(10, 10);
        for p in img.pixels_mut() {
            *p = image::Rgba([255, 0, 0, 255]);
        }
        let palette = vec![[255, 0, 0, 255]];
        let indexed = vec![0u8; 100];
        let result = resample(&img, &vec![0, 10], &vec![0, 10], &palette, &indexed).unwrap();
        assert_eq!(result.dimensions(), (1, 1));
        assert_eq!(result.get_pixel(0, 0), &image::Rgba([255, 0, 0, 255]));
    }
}
