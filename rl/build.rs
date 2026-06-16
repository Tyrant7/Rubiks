use std::path::Path;

fn main() {
    // Force the linker to include the CUDA warp_size symbol which would
    // otherwise be dropped by the Windows linker, causing CUDA to silently
    // fail to initialize at runtime.
    match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("windows") => {
            println!("cargo:rustc-link-arg=/INCLUDE:?warp_size@cuda@at@@YAHXZ");
        }
        Ok("linux") => {
            if let Ok(libtorch) = std::env::var("LIBTORCH") {
                let lib_dir = Path::new(&libtorch).join("lib");
                if lib_dir.join("libtorch_cuda.so").exists() {
                    println!("cargo:rustc-link-search=native={}", lib_dir.display());
                    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
                    println!("cargo:rustc-link-arg=-ltorch_cuda");
                    println!("cargo:rustc-link-arg=-lc10_cuda");
                    println!("cargo:rustc-link-arg=-Wl,-rpath={}", lib_dir.display());
                }
            }
        }
        _ => {}
    }
}
