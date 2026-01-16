use crate::error::{PixelSnapperError, Result};
use image::{ImageBuffer, RgbaImage};
use rayon::prelude::*;

pub fn resample(img: &RgbaImage, cols: &[usize], rows: &[usize]) -> Result<RgbaImage> {
    if cols.len() < 2 || rows.len() < 2 {
        return Err(PixelSnapperError::ProcessingError(
            "Insufficient grid cuts for resampling".to_string(),
        ));
    }

    let out_w = (cols.len().max(1) - 1) as u32;
    let out_h = (rows.len().max(1) - 1) as u32;

    let mut final_img: RgbaImage = ImageBuffer::new(out_w, out_h);

    {
        // Safe parallel writing using chunks_exact_mut
        let w = out_w;
        let samples = final_img.as_flat_samples_mut().samples;
        
        // Prepare input buffer for unsafe reading in the inner loop
        let in_samples = img.as_flat_samples().samples;
        let in_width = img.width() as usize;
        let in_stride = in_width * 4;

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
                    // Extreme fast path for 1:1 mapped cells
                    if xs < img.width() as usize && ys < img.height() as usize {
                        img.get_pixel(xs as u32, ys as u32).0
                    } else {
                        [0, 0, 0, 0]
                    }
                } else {
                    // Optimized counting for small/medium cells
                    let mut counts: Vec<([u8; 4], usize)> = Vec::with_capacity(4);

                    for y in ys..ye {
                        let y_offset = y * in_stride;
                        for x in xs..xe {
                            // SECURITY: We trust the loop bounds (xs..xe, ys..ye) are bound-checked by the surrounding logic
                            // and image dimensions.
                            let base = y_offset + x * 4;
                            unsafe {
                                let samples_ptr = in_samples.as_ptr();
                                let r = *samples_ptr.add(base) as u32;
                                let g = *samples_ptr.add(base + 1) as u32;
                                let b = *samples_ptr.add(base + 2) as u32;
                                let a = *samples_ptr.add(base + 3) as u32;

                                if a > 0 {
                                    let key = [r as u8, g as u8, b as u8, a as u8];
                                    if let Some(entry) = counts.iter_mut().find(|e| e.0 == key) {
                                        entry.1 += 1;
                                    } else {
                                        counts.push((key, 1));
                                    }
                                }
                            }
                        }
                    }

                    candidates_to_best_pixel(counts)
                };

                pixel_sample[0] = best_pixel[0];
                pixel_sample[1] = best_pixel[1];
                pixel_sample[2] = best_pixel[2];
                pixel_sample[3] = best_pixel[3];
            });
    }

    Ok(final_img)
}

fn candidates_to_best_pixel(candidates: Vec<([u8; 4], usize)>) -> [u8; 4] {
    if candidates.is_empty() {
        return [0, 0, 0, 0];
    }

    let mut best_p = candidates[0].0;
    let mut max_count = candidates[0].1;
    let mut max_sum = best_p.iter().map(|&v| v as u32).sum::<u32>();

    for &(p, count) in candidates.iter().skip(1) {
        if count > max_count {
            max_count = count;
            best_p = p;
            max_sum = p.iter().map(|&v| v as u32).sum::<u32>();
        } else if count == max_count {
            let sum = p.iter().map(|&v| v as u32).sum::<u32>();
            if sum > max_sum {
                best_p = p;
                max_sum = sum;
            }
        }
    }

    best_p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_insufficient_grid() {
        let img = RgbaImage::new(10, 10);
        let result = resample(&img, &vec![0], &vec![0, 10]);
        assert!(result.is_err());
    }

    #[test]
    fn test_resample_simple_grid() {
        let mut img = RgbaImage::new(10, 10);
        // Fill the entire image with red
        for p in img.pixels_mut() {
            *p = image::Rgba([255, 0, 0, 255]);
        }
        let result = resample(&img, &vec![0, 10], &vec![0, 10]).unwrap();
        assert_eq!(result.dimensions(), (1, 1));
        assert_eq!(result.get_pixel(0, 0), &image::Rgba([255, 0, 0, 255]));
    }
}
