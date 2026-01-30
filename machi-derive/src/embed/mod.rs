//! Implementation of the `#[derive(Embed)]` macro.
//!
//! This module handles the expansion of the `Embed` derive macro,
//! which generates implementations of the `machi::embedding::Embed` trait.
//!
//! # Submodules
//!
//! - [`fields`] - Field attribute parsing and handling

mod fields;

use proc_macro2::TokenStream;
use quote::{quote, quote_spanned};
use syn::DataStruct;

use fields::{add_struct_bounds, custom_embed_fields, simple_embed_fields};

/// The attribute name used for embedding fields.
pub(crate) const EMBED_ATTR: &str = "embed";

/// Expands the `#[derive(Embed)]` macro for a given input.
///
/// This function generates an implementation of the `Embed` trait that calls
/// `embed()` on all fields marked with `#[embed]` or `#[embed(embed_with = "...")]`.
pub(crate) fn expand(input: &mut syn::DeriveInput) -> syn::Result<TokenStream> {
    let name = &input.ident;
    let name_span = name.span();
    let data = &input.data;
    let generics = &mut input.generics;

    let body = match data {
        syn::Data::Struct(data_struct) => expand_struct(data_struct, generics, name)?,
        syn::Data::Enum(_) => {
            return Err(syn::Error::new_spanned(
                input,
                "Embed derive macro cannot be used on enums. Consider implementing Embed manually.",
            ));
        }
        syn::Data::Union(_) => {
            return Err(syn::Error::new_spanned(
                input,
                "Embed derive macro cannot be used on unions.",
            ));
        }
    };

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    Ok(quote_spanned! {name_span=>
        impl #impl_generics ::machi::embedding::Embed for #name #ty_generics #where_clause {
            fn embed(
                &self,
                embedder: &mut ::machi::embedding::TextEmbedder,
            ) -> ::core::result::Result<(), ::machi::embedding::EmbedError> {
                #body
                Ok(())
            }
        }
    })
}

/// Expand embedding implementation for a struct.
fn expand_struct(
    data_struct: &DataStruct,
    generics: &mut syn::Generics,
    name: &syn::Ident,
) -> syn::Result<TokenStream> {
    // Process simple #[embed] fields
    let simple_fields: Vec<_> = simple_embed_fields(data_struct)
        .map(|field| {
            add_struct_bounds(generics, &field.ty);
            let field_name = &field.ident;
            quote!(self.#field_name)
        })
        .collect();

    // Process #[embed(embed_with = "...")] fields
    let custom_fields: Vec<_> = custom_embed_fields(data_struct)?
        .into_iter()
        .map(|(field, func_path)| {
            let field_name = &field.ident;
            quote!(#func_path(embedder, self.#field_name.clone())?)
        })
        .collect();

    // Require at least one embedded field
    if simple_fields.is_empty() && custom_fields.is_empty() {
        return Err(syn::Error::new_spanned(
            name,
            "Embed derive requires at least one field tagged with #[embed] or #[embed(embed_with = \"...\")]",
        ));
    }

    Ok(quote_spanned! {name.span()=>
        #(::machi::embedding::Embed::embed(&#simple_fields, embedder)?;)*
        #(#custom_fields;)*
    })
}
