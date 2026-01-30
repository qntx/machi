//! Implementation of the `#[machi_tool]` attribute macro.
//!
//! This module transforms functions into tool implementations that can be used
//! with Machi agents.

use convert_case::{Case, Casing};
use proc_macro2::TokenStream;
use quote::{format_ident, quote, quote_spanned};
use std::collections::HashMap;
use syn::{
    Expr, ExprLit, FnArg, Ident, ItemFn, Lit, Meta, Pat, PathArguments, ReturnType, Token, Type,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
};

/// Parsed arguments from the `#[machi_tool(...)]` attribute.
#[derive(Default)]
pub(crate) struct ToolMacroArgs {
    pub description: Option<String>,
    pub param_descriptions: HashMap<String, String>,
    pub required: Vec<String>,
}

impl Parse for ToolMacroArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut result = Self::default();

        if input.is_empty() {
            return Ok(result);
        }

        let meta_list: Punctuated<Meta, Token![,]> = Punctuated::parse_terminated(input)?;

        for meta in meta_list {
            result.parse_meta_item(meta)?;
        }

        Ok(result)
    }
}

impl ToolMacroArgs {
    /// Parse a single meta item from the attribute arguments.
    fn parse_meta_item(&mut self, meta: Meta) -> syn::Result<()> {
        match meta {
            Meta::NameValue(nv) => {
                let ident = nv
                    .path
                    .get_ident()
                    .ok_or_else(|| syn::Error::new_spanned(&nv.path, "expected identifier"))?;

                if ident == "description" {
                    self.description = Some(extract_string_lit(&nv.value)?);
                }
                // Silently ignore unknown name-value pairs for forward compatibility
            }
            Meta::List(list) if list.path.is_ident("params") => {
                self.parse_params_list(&list)?;
            }
            Meta::List(list) if list.path.is_ident("required") => {
                self.parse_required_list(&list)?;
            }
            _ => {
                // Silently ignore unknown meta items for forward compatibility
            }
        }
        Ok(())
    }

    /// Parse the `params(...)` nested list.
    fn parse_params_list(&mut self, list: &syn::MetaList) -> syn::Result<()> {
        let nested: Punctuated<Meta, Token![,]> =
            list.parse_args_with(Punctuated::parse_terminated)?;

        for meta in nested {
            if let Meta::NameValue(nv) = meta {
                let param_name = nv
                    .path
                    .get_ident()
                    .ok_or_else(|| syn::Error::new_spanned(&nv.path, "expected parameter name"))?
                    .to_string();
                let description = extract_string_lit(&nv.value)?;
                self.param_descriptions.insert(param_name, description);
            }
        }
        Ok(())
    }

    /// Parse the `required(...)` nested list.
    fn parse_required_list(&mut self, list: &syn::MetaList) -> syn::Result<()> {
        let idents: Punctuated<Ident, Token![,]> =
            list.parse_args_with(Punctuated::parse_terminated)?;

        self.required = idents.into_iter().map(|id| id.to_string()).collect();
        Ok(())
    }
}

/// Extract a string literal from an expression.
fn extract_string_lit(expr: &Expr) -> syn::Result<String> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Str(lit_str),
            ..
        }) => Ok(lit_str.value()),
        _ => Err(syn::Error::new_spanned(expr, "expected string literal")),
    }
}

/// Information extracted from a function's return type.
struct ReturnTypeInfo {
    output_type: TokenStream,
    error_type: TokenStream,
}

/// Extract Output and Error types from a `Result<T, E>` return type.
fn extract_result_types(return_type: &ReturnType) -> syn::Result<ReturnTypeInfo> {
    let ReturnType::Type(_, ty) = return_type else {
        return Err(syn::Error::new_spanned(
            return_type,
            "function must have a return type of `Result<T, E>`",
        ));
    };

    let Type::Path(type_path) = ty.as_ref() else {
        return Err(syn::Error::new_spanned(
            ty,
            "return type must be a path type (e.g., `Result<T, E>`)",
        ));
    };

    let last_segment = type_path
        .path
        .segments
        .last()
        .ok_or_else(|| syn::Error::new_spanned(&type_path.path, "invalid return type path"))?;

    if last_segment.ident != "Result" {
        return Err(syn::Error::new_spanned(
            &last_segment.ident,
            "return type must be `Result<T, E>`",
        ));
    }

    let PathArguments::AngleBracketed(args) = &last_segment.arguments else {
        return Err(syn::Error::new_spanned(
            &last_segment.arguments,
            "expected angle bracketed type parameters for Result",
        ));
    };

    if args.args.len() != 2 {
        return Err(syn::Error::new_spanned(
            args,
            "Result must have exactly two type parameters: Result<T, E>",
        ));
    }

    let output = &args.args[0];
    let error = &args.args[1];

    Ok(ReturnTypeInfo {
        output_type: quote!(#output),
        error_type: quote!(#error),
    })
}

/// Information about a single function parameter.
struct ParamInfo<'a> {
    name: &'a Ident,
    ty: &'a Type,
    description: String,
    json_type: TokenStream,
}

/// Extract parameter information from function arguments.
fn extract_params<'a>(
    inputs: impl Iterator<Item = &'a FnArg>,
    param_descriptions: &HashMap<String, String>,
) -> Vec<ParamInfo<'a>> {
    inputs
        .filter_map(|arg| {
            let FnArg::Typed(pat_type) = arg else {
                return None;
            };
            let Pat::Ident(param_ident) = pat_type.pat.as_ref() else {
                return None;
            };

            let name = &param_ident.ident;
            let name_str = name.to_string();
            let ty = pat_type.ty.as_ref();
            let description = param_descriptions
                .get(&name_str)
                .cloned()
                .unwrap_or_else(|| format!("Parameter {name_str}"));
            let json_type = rust_type_to_json_schema(ty);

            Some(ParamInfo {
                name,
                ty,
                description,
                json_type,
            })
        })
        .collect()
}

/// Convert a Rust type to a JSON schema type representation.
fn rust_type_to_json_schema(ty: &Type) -> TokenStream {
    let Type::Path(type_path) = ty else {
        return quote! { "type": "object" };
    };

    let Some(segment) = type_path.path.segments.first() else {
        return quote! { "type": "object" };
    };

    let type_name = segment.ident.to_string();

    // Handle Vec<T> types
    if type_name == "Vec" {
        if let PathArguments::AngleBracketed(args) = &segment.arguments {
            if let Some(syn::GenericArgument::Type(inner_type)) = args.args.first() {
                let inner_json_type = rust_type_to_json_schema(inner_type);
                return quote! {
                    "type": "array",
                    "items": { #inner_json_type }
                };
            }
        }
        return quote! { "type": "array" };
    }

    // Handle Option<T> types
    if type_name == "Option" {
        if let PathArguments::AngleBracketed(args) = &segment.arguments {
            if let Some(syn::GenericArgument::Type(inner_type)) = args.args.first() {
                let inner_json_type = rust_type_to_json_schema(inner_type);
                // For Option, we return the inner type (nullable is handled separately)
                return inner_json_type;
            }
        }
        return quote! { "type": "object" };
    }

    // Handle primitive types
    match type_name.as_str() {
        "i8" | "i16" | "i32" | "i64" | "i128" | "isize" | "u8" | "u16" | "u32" | "u64" | "u128"
        | "usize" => quote! { "type": "integer" },
        "f32" | "f64" => quote! { "type": "number" },
        "String" | "str" | "Cow" => quote! { "type": "string" },
        "bool" => quote! { "type": "boolean" },
        _ => quote! { "type": "object" },
    }
}

/// Main entry point for the `#[machi_tool]` macro expansion.
pub(crate) fn expand_machi_tool(args: ToolMacroArgs, input_fn: ItemFn) -> syn::Result<TokenStream> {
    let fn_name = &input_fn.sig.ident;
    let fn_name_str = fn_name.to_string();
    let fn_span = input_fn.sig.ident.span();
    let is_async = input_fn.sig.asyncness.is_some();

    // Extract return type information
    let return_info = extract_result_types(&input_fn.sig.output)?;

    // Generate struct names
    let struct_name = format_ident!("{}", fn_name_str.to_case(Case::Pascal));
    let params_struct_name = format_ident!("{}Parameters", struct_name);
    let static_name = format_ident!("{}", fn_name_str.to_uppercase());

    // Extract parameter information
    let params = extract_params(input_fn.sig.inputs.iter(), &args.param_descriptions);
    let param_names: Vec<_> = params.iter().map(|p| p.name).collect();
    let param_types: Vec<_> = params.iter().map(|p| p.ty).collect();
    let param_descriptions: Vec<_> = params.iter().map(|p| &p.description).collect();
    let json_types: Vec<_> = params.iter().map(|p| &p.json_type).collect();

    // Generate description
    let tool_description = match args.description {
        Some(desc) => quote! { #desc.to_string() },
        None => quote! { format!("Function to {}", Self::NAME) },
    };

    let required_args = &args.required;

    // Generate the call implementation based on async/sync
    let call_impl = if is_async {
        quote! {
            async fn call(&self, args: Self::Args) -> ::core::result::Result<Self::Output, Self::Error> {
                #fn_name(#(args.#param_names,)*).await
            }
        }
    } else {
        quote! {
            async fn call(&self, args: Self::Args) -> ::core::result::Result<Self::Output, Self::Error> {
                #fn_name(#(args.#param_names,)*)
            }
        }
    };

    let output_type = &return_info.output_type;
    let error_type = &return_info.error_type;

    // Generate the expanded code with proper spans for error messages
    let expanded = quote_spanned! {fn_span=>
        #[derive(::serde::Deserialize)]
        pub(crate) struct #params_struct_name {
            #(#param_names: #param_types,)*
        }

        #input_fn

        #[derive(Default)]
        pub(crate) struct #struct_name;

        impl ::machi::tool::Tool for #struct_name {
            const NAME: &'static str = #fn_name_str;

            type Args = #params_struct_name;
            type Output = #output_type;
            type Error = #error_type;

            fn name(&self) -> ::std::string::String {
                #fn_name_str.to_string()
            }

            async fn definition(&self, _prompt: ::std::string::String) -> ::machi::completion::ToolDefinition {
                let parameters = ::serde_json::json!({
                    "type": "object",
                    "properties": {
                        #(
                            stringify!(#param_names): {
                                #json_types,
                                "description": #param_descriptions
                            }
                        ),*
                    },
                    "required": [#(#required_args),*]
                });

                ::machi::completion::ToolDefinition {
                    name: #fn_name_str.to_string(),
                    description: #tool_description,
                    parameters,
                }
            }

            #call_impl
        }

        pub(crate) static #static_name: #struct_name = #struct_name;
    };

    Ok(expanded)
}
