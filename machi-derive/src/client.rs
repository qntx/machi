//! Implementation of the `#[derive(ProviderClient)]` macro.
//!
//! This module generates empty trait implementations for provider capabilities
//! that are not specified in the `features` attribute.

use deluxe::{ParseAttributes, ParseMetaItem};
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{DeriveInput, parse_macro_input};

/// Parsed attributes from `#[client(...)]`.
#[derive(ParseMetaItem, Default, ParseAttributes)]
#[deluxe(attributes(client))]
struct ClientAttr {
    /// List of features/capabilities that this provider supports.
    pub features: Option<Vec<String>>,
}

/// Known provider capability features and their corresponding trait names.
const KNOWN_FEATURES: &[(&str, &str)] = &[
    ("completion", "AsCompletion"),
    ("transcription", "AsTranscription"),
    ("embeddings", "AsEmbeddings"),
    ("image_generation", "AsImageGeneration"),
    ("audio_generation", "AsAudioGeneration"),
];

/// Main entry point for the `#[derive(ProviderClient)]` macro.
///
/// This generates empty implementations for capability traits that are NOT
/// listed in the `features` attribute, allowing the type system to track
/// which capabilities a provider supports.
pub(crate) fn provider_client(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let ident = &input.ident;

    // Parse the #[client(features = [...])] attribute
    let attrs = match ClientAttr::parse_attributes(&input.attrs) {
        Ok(attrs) => attrs,
        Err(e) => return e.into_compile_error().into(),
    };
    let enabled_features: Vec<String> = attrs.features.unwrap_or_default();

    // Generate empty impls for features NOT in the enabled list
    let impls = KNOWN_FEATURES
        .iter()
        .filter(|(feature, _)| !enabled_features.iter().any(|f| f == *feature))
        .map(|(_, trait_name)| {
            let trait_ident = format_ident!("{}", trait_name);
            quote! {
                impl ::machi::client::#trait_ident for #ident {}
            }
        });

    let output = quote! {
        #(#impls)*
    };

    output.into()
}
