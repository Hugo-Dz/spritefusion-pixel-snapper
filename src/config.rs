#[cfg(not(target_arch = "wasm32"))]
use clap::Parser;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
#[cfg_attr(not(target_arch = "wasm32"), derive(Parser))]
#[cfg_attr(not(target_arch = "wasm32"), command(author, version, about, long_about = None))]
pub struct Config {
    /// Number of colors for quantization
    #[cfg_attr(not(target_arch = "wasm32"), arg(short, long, default_value_t = 16))]
    pub k_colors: usize,

    #[cfg_attr(
        not(target_arch = "wasm32"),
        arg(long, hide = true, default_value_t = 42)
    )]
    pub k_seed: u64,

    /// Input image path
    #[cfg_attr(not(target_arch = "wasm32"), arg())]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(skip))]
    pub input_path: String,

    /// Output image path
    #[cfg_attr(not(target_arch = "wasm32"), arg())]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(skip))]
    pub output_path: String,

    #[cfg_attr(
        not(target_arch = "wasm32"),
        arg(long, hide = true, default_value_t = 15)
    )]
    pub max_kmeans_iterations: usize,

    #[cfg_attr(
        not(target_arch = "wasm32"),
        arg(long, hide = true, default_value_t = 0.2)
    )]
    pub peak_threshold_multiplier: f64,

    #[cfg_attr(
        not(target_arch = "wasm32"),
        arg(long, hide = true, default_value_t = 4)
    )]
    pub peak_distance_filter: usize,

    #[cfg_attr(
        not(target_arch = "wasm32"),
        arg(long, hide = true, default_value_t = 0.35)
    )]
    pub walker_search_window_ratio: f64,

    #[cfg_attr(
        not(target_arch = "wasm32"),
        arg(long, hide = true, default_value_t = 2.0)
    )]
    pub walker_min_search_window: f64,

    #[cfg_attr(
        not(target_arch = "wasm32"),
        arg(long, hide = true, default_value_t = 0.5)
    )]
    pub walker_strength_threshold: f64,

    #[cfg_attr(
        not(target_arch = "wasm32"),
        arg(long, hide = true, default_value_t = 4)
    )]
    pub min_cuts_per_axis: usize,

    #[cfg_attr(
        not(target_arch = "wasm32"),
        arg(long, hide = true, default_value_t = 64)
    )]
    pub fallback_target_segments: usize,

    #[cfg_attr(
        not(target_arch = "wasm32"),
        arg(long, hide = true, default_value_t = 1.8)
    )]
    pub max_step_ratio: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            k_colors: 16,
            k_seed: 42,
            input_path: String::new(),
            output_path: String::new(),
            max_kmeans_iterations: 15,
            peak_threshold_multiplier: 0.2,
            peak_distance_filter: 4,
            walker_search_window_ratio: 0.35,
            walker_min_search_window: 2.0,
            walker_strength_threshold: 0.5,
            min_cuts_per_axis: 4,
            fallback_target_segments: 64,
            max_step_ratio: 1.8,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn parse_args() -> Option<Config> {
    Some(Config::parse())
}
