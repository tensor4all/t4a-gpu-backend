use super::*;

fn limits() -> Limits {
    Limits {
        pending_bytes: 64,
        pending_records: 4,
        idle_pinned_bytes: 0,
        max_chunk_bytes: 32,
    }
}

#[test]
fn oversized_indivisible_request_is_refused_before_admission() {
    let retirement = Retirement::new(limits());
    match capacity(&retirement, 65, 1) {
        Err(Error::Backpressure { bytes: 65 }) => {}
        other => panic!("expected Backpressure, got {other:?}"),
    }
}

#[test]
fn record_budget_is_enforced() {
    let retirement = Retirement::new(limits());
    assert!(capacity(&retirement, 0, 5).is_err());
    assert!(capacity(&retirement, 0, 4).is_ok());
}

#[test]
fn empty_retirement_reports_zero() {
    let retirement = Retirement::new(limits());
    let stats = retirement.stats();
    assert_eq!(stats.pending_bytes, 0);
    assert_eq!(stats.pending_records, 0);
    assert_eq!(stats.idle_chunks, 0);
    assert_eq!(stats.abandoned_chunks, 0);
    assert!(retirement.broken().is_none());
}

#[test]
fn broken_state_blocks_admission() {
    let retirement = Retirement::new(limits());
    retirement.mark_broken("driver fault");
    match capacity(&retirement, 1, 1) {
        Err(Error::Broken { detail }) => assert_eq!(detail, "driver fault"),
        other => panic!("expected Broken, got {other:?}"),
    }
}

/// Reserve capacity through the admission view, as the runtime does.
fn capacity(retirement: &Retirement, bytes: usize, records: usize) -> Result<()> {
    retirement.during_admission(|admission| {
        admission.ensure_capacity(std::ptr::null_mut(), bytes, records)
    })
}
