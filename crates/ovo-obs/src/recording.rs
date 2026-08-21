//! In-memory metrics capture for tests and local debugging.

use std::sync::Mutex;

use crate::metrics::MetricsSink;

/// Thread-safe capture of all metric events.
#[derive(Debug, Default)]
pub struct RecordingMetrics {
    counters: Mutex<Vec<CounterEvent>>,
    histograms: Mutex<Vec<HistogramEvent>>,
    gauges: Mutex<Vec<GaugeEvent>>,
}

/// Captured counter sample.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CounterEvent {
    /// Series name.
    pub name: String,
    /// Increment.
    pub value: u64,
    /// Labels as `k=v` pairs.
    pub labels: Vec<(String, String)>,
}

/// Captured histogram sample.
#[derive(Debug, Clone, PartialEq)]
pub struct HistogramEvent {
    /// Series name.
    pub name: String,
    /// Observed value.
    pub value: f64,
    /// Labels.
    pub labels: Vec<(String, String)>,
}

/// Captured gauge sample.
#[derive(Debug, Clone, PartialEq)]
pub struct GaugeEvent {
    /// Series name.
    pub name: String,
    /// Gauge value.
    pub value: f64,
    /// Labels.
    pub labels: Vec<(String, String)>,
}

impl RecordingMetrics {
    /// Empty recorder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot of counter events.
    #[must_use]
    pub fn counters(&self) -> Vec<CounterEvent> {
        self.counters
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Snapshot of histogram events.
    #[must_use]
    pub fn histograms(&self) -> Vec<HistogramEvent> {
        self.histograms
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Sum of counter values for a series name.
    #[must_use]
    pub fn counter_sum(&self, name: &str) -> u64 {
        self.counters()
            .into_iter()
            .filter(|e| e.name == name)
            .map(|e| e.value)
            .sum()
    }

    /// Whether any event used this series name.
    #[must_use]
    pub fn saw(&self, name: &str) -> bool {
        self.counters().iter().any(|e| e.name == name)
            || self.histograms().iter().any(|e| e.name == name)
            || self
                .gauges
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .iter()
                .any(|e| e.name == name)
    }
}

impl MetricsSink for RecordingMetrics {
    fn counter(&self, name: &str, value: u64, labels: &[(&str, &str)]) {
        self.counters
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(CounterEvent {
                name: name.to_owned(),
                value,
                labels: owned_labels(labels),
            });
    }

    fn histogram(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        self.histograms
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(HistogramEvent {
                name: name.to_owned(),
                value,
                labels: owned_labels(labels),
            });
    }

    fn gauge(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        self.gauges
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(GaugeEvent {
                name: name.to_owned(),
                value,
                labels: owned_labels(labels),
            });
    }
}

fn owned_labels(labels: &[(&str, &str)]) -> Vec<(String, String)> {
    labels
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{METRIC_TURNS_TOTAL, record_turn};

    #[test]
    fn captures_turn() {
        let rec = RecordingMetrics::new();
        record_turn(&rec, "ok", 2, 5.0);
        assert!(rec.saw(METRIC_TURNS_TOTAL));
        assert_eq!(rec.counter_sum(METRIC_TURNS_TOTAL), 1);
        assert!(!rec.histograms().is_empty());
    }
}
