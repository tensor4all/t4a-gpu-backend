//! Bounds, introspection and clearing for the persistent artifact directory.
//!
//! Only files the cache itself names are ever read, counted or removed: published images
//! `v1-*.ptx` / `v1-*.cubin` and staging files `.v1-*.tmp`. A directory a caller points at
//! by mistake (a home directory, a shared scratch) therefore loses nothing else.
//!
//! Eviction is least-recently-used by modification time: publishing an image sets it, and a
//! read from the directory refreshes it, so the modules a workload keeps loading stay while
//! ones nobody has asked for since go first.

use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// Default bound on the bytes of images kept in a persistent artifact directory: 256 MiB.
///
/// It matches the in-memory artifact cache's default byte budget
/// ([`crate::Config::artifact_cache_bytes`]) and the 256 MiB proposed per artifact cache in the
/// runtime design. One compiled module is typically hundreds of KiB to a few MiB, so this holds
/// a whole kernel catalogue for a few toolchain versions. A caller changes it with
/// [`crate::ArtifactCache::with_dir_max_bytes`] or
/// [`crate::Runtime::set_artifact_dir_max_bytes`].
pub const DEFAULT_ARTIFACT_DIR_BYTES: u64 = 256 << 20;

/// A staging file older than this belongs to a writer that died before renaming it.
const STALE_STAGING: Duration = Duration::from_secs(60 * 60);

/// What a persistent artifact directory holds, or what an operation removed from it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ArtifactDirStats {
    /// Number of published images.
    pub entries: usize,
    /// Total size of those images in bytes.
    pub bytes: u64,
}

impl ArtifactDirStats {
    fn add(&mut self, bytes: u64) {
        self.entries += 1;
        self.bytes += bytes;
    }
}

/// One published image in the directory.
struct Entry {
    path: PathBuf,
    bytes: u64,
    used: SystemTime,
}

fn is_image(name: &str) -> bool {
    name.starts_with("v1-") && (name.ends_with(".ptx") || name.ends_with(".cubin"))
}

fn is_staging(name: &str) -> bool {
    name.starts_with(".v1-") && name.ends_with(".tmp")
}

/// Published images and staging files in `dir`; a missing directory holds neither.
fn scan(dir: &Path) -> io::Result<(Vec<Entry>, Vec<Entry>)> {
    let listing = match std::fs::read_dir(dir) {
        Ok(listing) => listing,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Ok((Vec::new(), Vec::new()))
        }
        Err(error) => return Err(error),
    };
    let mut images = Vec::new();
    let mut staging = Vec::new();
    for item in listing {
        // An entry that vanished under a concurrent trim is simply not there.
        let Ok(item) = item else { continue };
        let name = item.file_name();
        let Some(name) = name.to_str() else { continue };
        let target = if is_image(name) {
            &mut images
        } else if is_staging(name) {
            &mut staging
        } else {
            continue;
        };
        let Ok(metadata) = item.metadata() else {
            continue;
        };
        if !metadata.is_file() {
            continue;
        }
        target.push(Entry {
            path: item.path(),
            bytes: metadata.len(),
            used: metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH),
        });
    }
    Ok((images, staging))
}

/// Remove `path`; one that is already gone counts as removed by someone else, not by us.
pub(super) fn remove(path: &Path) -> io::Result<bool> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    }
}

/// Number and total size of the images in a persistent artifact directory.
///
/// A directory that does not exist yet holds nothing.
///
/// # Errors
///
/// The I/O error from listing a directory that exists but cannot be read.
pub fn artifact_dir_stats(dir: &Path) -> io::Result<ArtifactDirStats> {
    let (images, _) = scan(dir)?;
    let mut stats = ArtifactDirStats::default();
    for image in &images {
        stats.add(image.bytes);
    }
    Ok(stats)
}

/// Remove least-recently-used images until the directory holds at most `max_bytes`.
///
/// Also removes staging files a crashed writer left behind more than an hour ago. Returns the
/// images removed.
///
/// # Errors
///
/// The I/O error from listing the directory or removing an entry.
pub fn trim_artifact_dir(dir: &Path, max_bytes: u64) -> io::Result<ArtifactDirStats> {
    trim_except(dir, max_bytes, None)
}

/// [`trim_artifact_dir`], never removing `keep` (the image just published).
pub(super) fn trim_except(
    dir: &Path,
    max_bytes: u64,
    keep: Option<&Path>,
) -> io::Result<ArtifactDirStats> {
    let mut evicted = ArtifactDirStats::default();
    trim_into(dir, max_bytes, keep, &mut evicted, remove)?;
    Ok(evicted)
}

/// [`trim_except`] with the removal injected, adding each image to `evicted` as soon as it
/// is gone, so a trim that fails part way still leaves in `evicted` what it removed before.
pub(super) fn trim_into(
    dir: &Path,
    max_bytes: u64,
    keep: Option<&Path>,
    evicted: &mut ArtifactDirStats,
    mut remove: impl FnMut(&Path) -> io::Result<bool>,
) -> io::Result<()> {
    let (mut images, staging) = scan(dir)?;
    let now = SystemTime::now();
    for stale in staging {
        if now.duration_since(stale.used).unwrap_or_default() > STALE_STAGING {
            remove(&stale.path)?;
        }
    }
    let mut held: u64 = images.iter().map(|image| image.bytes).sum();
    if held <= max_bytes {
        return Ok(());
    }
    // Oldest use first; equal times fall back to the name so the order is deterministic.
    images.sort_by(|a, b| a.used.cmp(&b.used).then_with(|| a.path.cmp(&b.path)));
    for image in images {
        if held <= max_bytes {
            break;
        }
        if keep == Some(image.path.as_path()) {
            continue;
        }
        if remove(&image.path)? {
            evicted.add(image.bytes);
        }
        held = held.saturating_sub(image.bytes);
    }
    Ok(())
}

/// Remove every image and staging file from a persistent artifact directory.
///
/// Files the cache did not name are left alone, and so is the directory itself. Returns the
/// images removed.
///
/// # Errors
///
/// The I/O error from listing the directory or removing an entry.
pub fn clear_artifact_dir(dir: &Path) -> io::Result<ArtifactDirStats> {
    let (images, staging) = scan(dir)?;
    for stale in staging {
        remove(&stale.path)?;
    }
    let mut removed = ArtifactDirStats::default();
    for image in images {
        if remove(&image.path)? {
            removed.add(image.bytes);
        }
    }
    Ok(removed)
}

/// Record a use of `path` so eviction keeps it; best effort, as a read-only cache cannot.
pub(super) fn touch(path: &Path) {
    let _ = std::fs::File::options()
        .write(true)
        .open(path)
        .and_then(|file| file.set_modified(SystemTime::now()));
}
