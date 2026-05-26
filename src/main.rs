use fuser::{Config, MountOption};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Minimal CLI: rfs <mountpoint> [--image <path>]
    let mut image: Option<String> = None;
    let mut mountpoint: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--image" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--image requires a path");
                    std::process::exit(2);
                }
                image = Some(args[i].clone());
            }
            "-h" | "--help" => {
                print_usage(&args[0]);
                std::process::exit(0);
            }
            other if !other.starts_with('-') && mountpoint.is_none() => {
                mountpoint = Some(other.to_string());
            }
            other => {
                eprintln!("unknown arg: {other}");
                print_usage(&args[0]);
                std::process::exit(2);
            }
        }
        i += 1;
    }
    let Some(mountpoint) = mountpoint else {
        print_usage(&args[0]);
        std::process::exit(2);
    };

    let mut config = Config::default();
    config.mount_options = vec![
        MountOption::FSName("rfs".to_string()),
        MountOption::DefaultPermissions,
    ];

    let fs = match image {
        None => rfs::fuse::FuseFs::new(),
        Some(p) => {
            let path = std::path::PathBuf::from(p);
            let result = if path.exists() {
                rfs::fuse::FuseFs::open_image(&path)
            } else {
                rfs::fuse::FuseFs::create_image(&path)
            };
            match result {
                Ok(fs) => fs,
                Err(e) => {
                    eprintln!("image open/create failed: {e}");
                    std::process::exit(1);
                }
            }
        }
    };
    if let Err(e) = fuser::mount2(fs, mountpoint, &config) {
        eprintln!("mount failed: {e}");
        std::process::exit(1);
    }
}

fn print_usage(prog: &str) {
    eprintln!("usage: {prog} <mountpoint> [--image <path>]");
    eprintln!();
    eprintln!("options:");
    eprintln!("  --image <path>   back the filesystem with an image file at <path>");
    eprintln!("                   (created if missing, opened+verified if present).");
    eprintln!("                   without this flag the filesystem is in-memory only");
    eprintln!("                   and all data is lost on unmount.");
}
