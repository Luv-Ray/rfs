use fuser::{Config, MountOption};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mountpoint = match args.get(1) {
        Some(p) => p,
        None => {
            eprintln!(
                "usage: {} <mountpoint>",
                args.first().map(String::as_str).unwrap_or("rfs")
            );
            std::process::exit(2);
        }
    };
    let mut config = Config::default();
    config.mount_options = vec![
        MountOption::FSName("rfs".to_string()),
        MountOption::DefaultPermissions,
    ];
    let fs = rfs::fuse::FuseFs::new();
    if let Err(e) = fuser::mount2(fs, mountpoint, &config) {
        eprintln!("mount failed: {e}");
        std::process::exit(1);
    }
}
