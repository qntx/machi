//! Field attribute parsing for the Embed derive macro.
//!
//! This module handles both simple `#[embed]` attributes and
//! custom `#[embed(embed_with = "...")]` attributes.

use quote::ToTokens;
use syn::{Attribute, DataStruct, ExprPath, Meta, meta::ParseNestedMeta, parse_quote};

use super::EMBED_ATTR;

/// The attribute key for specifying custom embedding functions.
const EMBED_WITH_KEY: &str = "embed_with";

// =============================================================================
// Simple #[embed] Fields
// =============================================================================

/// Finds and returns fields with simple `#[embed]` attribute tags.
///
/// This filters out fields that use `#[embed(embed_with = "...")]` syntax.
pub(crate) fn simple_embed_fields(data_struct: &DataStruct) -> impl Iterator<Item = &syn::Field> {
    data_struct.fields.iter().filter(has_simple_embed_attr)
}

/// Check if a field has a simple `#[embed]` attribute (not `#[embed(...)]`).
fn has_simple_embed_attr(field: &&syn::Field) -> bool {
    field.attrs.iter().any(|attr| {
        matches!(
            attr,
            Attribute {
                meta: Meta::Path(path),
                ..
            } if path.is_ident(EMBED_ATTR)
        )
    })
}

/// Adds trait bounds to the where clause for embedded fields.
pub(crate) fn add_struct_bounds(generics: &mut syn::Generics, field_type: &syn::Type) {
    let where_clause = generics.make_where_clause();
    where_clause.predicates.push(parse_quote! {
        #field_type: ::machi::embedding::Embed
    });
}

// =============================================================================
// Custom #[embed(embed_with = "...")] Fields
// =============================================================================

/// Finds and returns fields with `#[embed(embed_with = "...")]` attribute tags.
///
/// Returns a vector of tuples containing the field and its custom embedding function path.
pub(crate) fn custom_embed_fields(
    data_struct: &DataStruct,
) -> syn::Result<Vec<(&syn::Field, ExprPath)>> {
    data_struct
        .fields
        .iter()
        .filter_map(|field| extract_custom_info(field).transpose())
        .collect()
}

/// Extract custom embedding info from a single field.
fn extract_custom_info(field: &syn::Field) -> syn::Result<Option<(&syn::Field, ExprPath)>> {
    for attr in &field.attrs {
        if let Some(path) = parse_custom_attr(attr)? {
            return Ok(Some((field, path)));
        }
    }
    Ok(None)
}

/// Parse a custom embed attribute, returning the function path if valid.
fn parse_custom_attr(attr: &Attribute) -> syn::Result<Option<ExprPath>> {
    // Must be a list attribute: #[embed(...)]
    let Meta::List(meta) = &attr.meta else {
        return Ok(None);
    };
    if meta.tokens.is_empty() || !attr.path().is_ident(EMBED_ATTR) {
        return Ok(None);
    }

    // Parse and validate the nested content
    let mut result: Option<ExprPath> = None;

    attr.parse_nested_meta(|meta| {
        // Validate the attribute name
        if !meta.path.is_ident(EMBED_WITH_KEY) {
            let path = meta.path.to_token_stream().to_string().replace(' ', "");
            return Err(syn::Error::new_spanned(
                meta.path,
                format!("unknown embedding field attribute `{path}`"),
            ));
        }

        // Parse the function path
        result = Some(parse_embed_with_value(&meta)?);
        Ok(())
    })?;

    Ok(result)
}

/// Parse the value of `embed_with = "..."`.
fn parse_embed_with_value(meta: &ParseNestedMeta<'_>) -> syn::Result<ExprPath> {
    let expr = meta.value()?.parse::<syn::Expr>()?;

    // Unwrap any grouping expressions
    let mut value = &expr;
    while let syn::Expr::Group(e) = value {
        value = &e.expr;
    }

    // Extract and validate the string literal
    let syn::Expr::Lit(syn::ExprLit {
        lit: syn::Lit::Str(lit_str),
        ..
    }) = value
    else {
        return Err(syn::Error::new_spanned(
            value,
            format!("expected `{EMBED_WITH_KEY}` to be a string: `{EMBED_WITH_KEY} = \"...\"`"),
        ));
    };

    if !lit_str.suffix().is_empty() {
        return Err(syn::Error::new_spanned(
            lit_str,
            format!("unexpected suffix `{}` on string literal", lit_str.suffix()),
        ));
    }

    lit_str.parse()
}
