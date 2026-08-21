//! Prometheus text exposition for captured metrics (zero external deps).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::sync::Mutex;

use crate::metrics::MetricsSink;

type LabelSet = Vec<(String, String)>;
type SeriesKey = (String, LabelSet);

/// Metrics sink that accumulates samples and can render Prometheus text format 0.0.4.
#[derive(Debug, Default)]
pub struct PrometheusRecorder {
    counters: Mutex<BTreeMap<SeriesKey, u64>>,
    histograms: Mutex<BTreeMap<SeriesKey, Vec<f64>>>,
    gauges: Mutex<BTreeMap<SeriesKey, f64>>,
}

impl PrometheusRecorder {
    /// Empty recorder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Render Prometheus exposition text (sorted, one `# TYPE` per metric name).
    #[must_use]
    pub fn render(&self) -> String {
        let mut out = String::new();
        let mut typed = BTreeSet::new();

        render_counters(&self.counters, &mut out, &mut typed);
        render_gauges(&self.gauges, &mut out, &mut typed);
        render_histograms(&self.histograms, &mut out, &mut typed);
        out
    }

    /// Series names that appear in the current capture (counters/histograms/gauges).
    #[must_use]
    pub fn series_names(&self) -> BTreeSet<String> {
        let mut names = BTreeSet::new();
        for (n, _) in self
            .counters
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .keys()
        {
            names.insert(n.clone());
        }
        for (n, _) in self
            .histograms
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .keys()
        {
            names.insert(n.clone());
        }
        for (n, _) in self
            .gauges
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .keys()
        {
            names.insert(n.clone());
        }
        names
    }
}

fn render_counters(
    counters: &Mutex<BTreeMap<SeriesKey, u64>>,
    out: &mut String,
    typed: &mut BTreeSet<String>,
) {
    let guard = counters
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for ((name, labels), value) in guard.iter() {
        emit_type_once(out, typed, name, "counter", "Ovo kernel counter");
        let _ = writeln!(out, "{name}{} {value}", format_labels(labels));
    }
}

fn render_gauges(
    gauges: &Mutex<BTreeMap<SeriesKey, f64>>,
    out: &mut String,
    typed: &mut BTreeSet<String>,
) {
    let guard = gauges
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for ((name, labels), value) in guard.iter() {
        emit_type_once(out, typed, name, "gauge", "Ovo kernel gauge");
        let _ = writeln!(out, "{name}{} {value}", format_labels(labels));
    }
}

fn render_histograms(
    histograms: &Mutex<BTreeMap<SeriesKey, Vec<f64>>>,
    out: &mut String,
    typed: &mut BTreeSet<String>,
) {
    let guard = histograms
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for ((name, labels), samples) in guard.iter() {
        emit_type_once(out, typed, name, "summary", "Ovo kernel summary");
        let count = samples.len();
        let sum: f64 = samples.iter().sum();
        let labs = format_labels(labels);
        let _ = writeln!(out, "{name}_count{labs} {count}");
        let _ = writeln!(out, "{name}_sum{labs} {sum}");
    }
}

fn emit_type_once(
    out: &mut String,
    typed: &mut BTreeSet<String>,
    name: &str,
    type_name: &str,
    help: &str,
) {
    if typed.insert(name.to_owned()) {
        let _ = writeln!(out, "# HELP {name} {help}");
        let _ = writeln!(out, "# TYPE {name} {type_name}");
    }
}

impl MetricsSink for PrometheusRecorder {
    fn counter(&self, name: &str, value: u64, labels: &[(&str, &str)]) {
        let key = (name.to_owned(), owned_labels(labels));
        *self
            .counters
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(key)
            .or_insert(0) += value;
    }

    fn histogram(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        let key = (name.to_owned(), owned_labels(labels));
        self.histograms
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(key)
            .or_default()
            .push(value);
    }

    fn gauge(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        let key = (name.to_owned(), owned_labels(labels));
        self.gauges
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(key, value);
    }
}

fn owned_labels(labels: &[(&str, &str)]) -> Vec<(String, String)> {
    labels
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect()
}

fn format_labels(labels: &[(String, String)]) -> String {
    if labels.is_empty() {
        return String::new();
    }
    let parts: Vec<String> = labels
        .iter()
        .map(|(k, v)| format!("{k}=\"{}\"", escape_label(v)))
        .collect();
    format!("{{{}}}", parts.join(","))
}

fn escape_label(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{
        METRIC_SAMPLE_DURATION_MS, METRIC_TURNS_TOTAL, emit_catalogue_smoke, record_turn,
        required_metric_names,
    };

    #[test]
    fn renders_counter() {
        let p = PrometheusRecorder::new();
        record_turn(&p, "ok", 1, 2.0);
        let text = p.render();
        assert!(text.contains(METRIC_TURNS_TOTAL), "{text}");
        assert!(text.contains("status=\"ok\""), "{text}");
        assert!(text.contains("# TYPE"), "{text}");
        assert!(text.contains("# HELP"), "{text}");
        let type_lines = text
            .lines()
            .filter(|l| *l == format!("# TYPE {METRIC_TURNS_TOTAL} counter"))
            .count();
        assert_eq!(type_lines, 1, "{text}");
    }

    #[test]
    fn golden_export_covers_catalogue() {
        let p = PrometheusRecorder::new();
        emit_catalogue_smoke(&p);
        let text = p.render();
        let names = p.series_names();
        for required in required_metric_names() {
            assert!(
                names.contains(*required),
                "export missing {required}\n{text}"
            );
        }
        assert!(
            text.contains(&format!("{METRIC_SAMPLE_DURATION_MS}_count")),
            "{text}"
        );
        assert!(text.contains("# HELP"));
        assert!(text.contains("# TYPE"));
        assert!(text.contains("status=\"ok\"") || text.contains("outcome=\"completed\""));
    }

    #[test]
    fn escapes_label_values() {
        let p = PrometheusRecorder::new();
        p.counter("ovo_test", 1, &[("path", r#"a"b\c"#)]);
        let text = p.render();
        assert!(text.contains(r#"path="a\"b\\c""#), "{text}");
    }
}
