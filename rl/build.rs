fn main() {
    // Force the linker to include the CUDA warp_size symbol which would
    // otherwise be dropped by the Windows linker, causing CUDA to silently
    // fail to initialize at runtime.
    println!("cargo:rustc-link-arg=/INCLUDE:?warp_size@cuda@at@@YAHXZ");
}
