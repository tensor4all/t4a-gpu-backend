//! The raw consumer seam.
//!
//! Consumers that need an encoder type this runtime does not provide (a blit, a
//! vendor object from MetalPerformanceShaders, a hand-built pipeline) get the
//! command buffer after the open compute encoder is closed, encode into it
//! themselves, and let the runtime's `commit` submit it. The unsafe caller owns the
//! encoders it opens and the ABIs it asserts, exactly as with the CUDA runtime's raw
//! scope.

use std::ffi::c_void;
use std::sync::Arc;

use t4a_metal_sys::objc::Id;

use crate::context::ContextHolder;
use crate::error::{Error, Result};
use crate::memory::Buffer;

/// Borrowed access to the submission the runtime is building.
pub struct RawScope<'a> {
    context: &'a Arc<ContextHolder>,
    command_buffer: Id,
}

impl RawScope<'_> {
    pub(crate) fn new(context: &Arc<ContextHolder>, command_buffer: Id) -> RawScope<'_> {
        RawScope {
            context,
            command_buffer,
        }
    }

    /// The `MTLCommandBuffer` to encode into.
    ///
    /// Every compute encoder the runtime had open is already closed, so the caller
    /// may open any encoder type.
    pub fn command_buffer(&self) -> *mut c_void {
        self.command_buffer
    }

    /// The `MTLDevice` the submission targets.
    pub fn device(&self) -> *mut c_void {
        self.context.device()
    }

    /// The allocation-domain identity this submission belongs to.
    pub fn domain_id(&self) -> u64 {
        self.context.domain_id()
    }

    /// The `MTLBuffer` behind `buffer`, after the domain check.
    ///
    /// # Errors
    ///
    /// [`Error::ForeignDomain`] when the buffer belongs to another runtime.
    pub fn buffer(&self, buffer: &Buffer) -> Result<*mut c_void> {
        if !Arc::ptr_eq(buffer.context(), self.context) {
            return Err(Error::ForeignDomain);
        }
        Ok(buffer.handle())
    }
}

impl std::fmt::Debug for RawScope<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawScope")
            .field("domain_id", &self.context.domain_id())
            .finish_non_exhaustive()
    }
}
