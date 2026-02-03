//! Tool macro implementation.
//!
//! This module contains the implementation of the `#[tool]` attribute macro
//! that transforms functions into Tool implementations.

use convert_case::{Case, Casing};
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use std::collections::HashMap;
use syn::{
    Expr, ExprLit, FnArg, Ident, ItemFn, Lit, Meta, Pat, PathArguments, ReturnType, Token, Type,
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
};

/// Parsed macro arguments for the `#[tool]` attribute.
struct ToolArgs {
    /// Tool description
    description: Option<String>,
    /// Parameter descriptions
    param_descriptions: HashMap<String, String>,
    /// Required parameters (non-optional)
    required: Vec<String>,
}

impl Parse for ToolArgs {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut description = None;
        let mut param_descriptions = HashMap::new();
        let mut required = Vec::new();

        if input.is_empty() {
            return Ok(Self {
                description,
                param_descriptions,
                required,
            });
        }

        let meta_list: Punctuated<Meta, Token![,]> = Punctuated::parse_terminated(input)?;

        for meta in meta_list {
            match meta {
                Meta::NameValue(nv) => {
                    let ident = nv.path.get_ident().map(ToString::to_string);
                    if let (
                        Some(ident_str),
                        Expr::Lit(ExprLit {
                            lit: Lit::Str(lit_str),
                            ..
                        }),
                    ) = (ident, nv.value)
                    {
                        if ident_str == "description" {
                            description = Some(lit_str.value());
                        }
                    }
                }
                Meta::List(list) if list.path.is_ident("params") => {
                    let nested: Punctuated<Meta, Token![,]> =
                        list.parse_args_with(Punctuated::parse_terminated)?;

                    for meta in nested {
                        if let Meta::NameValue(nv) = meta {
                            if let Expr::Lit(ExprLit {
                                lit: Lit::Str(lit_str),
                                ..
                            }) = nv.value
                            {
                                if let Some(param_name) = nv.path.get_ident() {
                                    param_descriptions
                                        .insert(param_name.to_string(), lit_str.value());
                                }
                            }
                        }
                    }
                }
                Meta::List(list) if list.path.is_ident("required") => {
                    let required_vars: Punctuated<Ident, Token![,]> =
                        list.parse_args_with(Punctuated::parse_terminated)?;

                    for var in required_vars {
                        required.push(var.to_string());
                    }
                }
                _ => {}
            }
        }

        Ok(Self {
            description,
            param_descriptions,
            required,
        })
    }
}

/// Get JSON schema type for a Rust type.
fn get_json_type(ty: &Type) -> TokenStream2 {
    match ty {
        Type::Path(type_path) => {
            let segment = &type_path.path.segments[0];
            let type_name = segment.ident.to_string();

            // Handle Vec types
            if type_name == "Vec" {
                if let PathArguments::AngleBracketed(args) = &segment.arguments {
                    if let Some(syn::GenericArgument::Type(inner_type)) = args.args.first() {
                        let inner_json_type = get_json_type(inner_type);
                        return quote! {
                            "type": "array",
                            "items": { #inner_json_type }
                        };
                    }
                }
                return quote! { "type": "array" };
            }

            // Handle Option types
            if type_name == "Option" {
                if let PathArguments::AngleBracketed(args) = &segment.arguments {
                    if let Some(syn::GenericArgument::Type(inner_type)) = args.args.first() {
                        let inner_json_type = get_json_type(inner_type);
                        return quote! {
                            #inner_json_type,
                            "nullable": true
                        };
                    }
                }
                return quote! { "type": "object", "nullable": true };
            }

            // Handle primitive types
            match type_name.as_str() {
                "i8" | "i16" | "i32" | "i64" | "i128" | "isize" | "u8" | "u16" | "u32" | "u64"
                | "u128" | "usize" => {
                    quote! { "type": "integer" }
                }
                "f32" | "f64" => {
                    quote! { "type": "number" }
                }
                "String" | "str" => {
                    quote! { "type": "string" }
                }
                "bool" => {
                    quote! { "type": "boolean" }
                }
                _ => {
                    quote! { "type": "object" }
                }
            }
        }
        Type::Reference(type_ref) => {
            // Handle &str
            if let Type::Path(inner) = &*type_ref.elem {
                if inner.path.is_ident("str") {
                    return quote! { "type": "string" };
                }
            }
            get_json_type(&type_ref.elem)
        }
        _ => {
            quote! { "type": "object" }
        }
    }
}

/// Check if a type is Option<T>
fn is_option_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.first() {
            return segment.ident == "Option";
        }
    }
    false
}

/// Main implementation of the tool macro.
pub fn tool_impl(args: TokenStream, input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as ToolArgs);
    let input_fn = parse_macro_input!(input as ItemFn);

    // Extract function details
    let fn_name = &input_fn.sig.ident;
    let fn_name_str = fn_name.to_string();
    let is_async = input_fn.sig.asyncness.is_some();
    let fn_visibility = &input_fn.vis;

    // Extract return type
    let return_type = &input_fn.sig.output;
    let (output_type, error_type) = extract_result_types(return_type);

    // Generate struct name (PascalCase)
    let struct_name = format_ident!("{}", fn_name_str.to_case(Case::Pascal));
    let params_struct_name = format_ident!("{}Args", struct_name);
    let static_name = format_ident!("{}", fn_name_str.to_uppercase());

    // Generate description
    let tool_description = match &args.description {
        Some(desc) => quote! { #desc.to_string() },
        None => quote! { format!("Tool function: {}", #fn_name_str) },
    };

    // Extract parameters
    let mut param_names = Vec::new();
    let mut param_types = Vec::new();
    let mut param_descriptions = Vec::new();
    let mut json_types = Vec::new();
    let mut required_params = Vec::new();

    for arg in &input_fn.sig.inputs {
        if let FnArg::Typed(pat_type) = arg {
            if let Pat::Ident(param_ident) = &*pat_type.pat {
                let param_name = &param_ident.ident;
                let param_name_str = param_name.to_string();
                let ty = &*pat_type.ty;

                let description = args
                    .param_descriptions
                    .get(&param_name_str)
                    .cloned()
                    .unwrap_or_else(|| format!("Parameter {param_name_str}"));

                // Determine if required (not Option and not in explicit required list override)
                let is_required = !is_option_type(ty) || args.required.contains(&param_name_str);

                if is_required {
                    required_params.push(param_name_str.clone());
                }

                param_names.push(param_name.clone());
                param_types.push(ty.clone());
                param_descriptions.push(description);
                json_types.push(get_json_type(ty));
            }
        }
    }

    // Generate call implementation
    let call_impl = if is_async {
        quote! {
            async fn call(
                &self,
                args: Self::Args,
            ) -> ::std::result::Result<Self::Output, Self::Error> {
                #fn_name(#(args.#param_names,)*).await
            }
        }
    } else {
        quote! {
            async fn call(
                &self,
                args: Self::Args,
            ) -> ::std::result::Result<Self::Output, Self::Error> {
                #fn_name(#(args.#param_names,)*)
            }
        }
    };

    let expanded = quote! {
        /// Arguments for the [`#struct_name`] tool.
        #[derive(Debug, Clone, ::serde::Deserialize, ::serde::Serialize)]
        #fn_visibility struct #params_struct_name {
            #(
                /// #param_descriptions
                pub #param_names: #param_types,
            )*
        }

        #input_fn

        /// Tool struct generated from function [`#fn_name`].
        #[derive(Debug, Clone, Copy, Default)]
        #fn_visibility struct #struct_name;

        impl ::machi::tool::Tool for #struct_name {
            const NAME: &'static str = #fn_name_str;

            type Args = #params_struct_name;
            type Output = #output_type;
            type Error = #error_type;

            fn name(&self) -> &'static str {
                #fn_name_str
            }

            fn description(&self) -> ::std::string::String {
                #tool_description
            }

            fn parameters_schema(&self) -> ::serde_json::Value {
                ::serde_json::json!({
                    "type": "object",
                    "properties": {
                        #(
                            stringify!(#param_names): {
                                #json_types,
                                "description": #param_descriptions
                            }
                        ),*
                    },
                    "required": [#(#required_params),*]
                })
            }

            #call_impl
        }

        /// Static instance of the [`#struct_name`] tool.
        #fn_visibility static #static_name: #struct_name = #struct_name;
    };

    TokenStream::from(expanded)
}

/// Extract Output and Error types from Result<T, E>.
fn extract_result_types(return_type: &ReturnType) -> (TokenStream2, TokenStream2) {
    match return_type {
        ReturnType::Type(_, ty) => {
            if let Type::Path(type_path) = &**ty {
                if let Some(last_segment) = type_path.path.segments.last() {
                    if last_segment.ident == "Result" {
                        if let PathArguments::AngleBracketed(args) = &last_segment.arguments {
                            if args.args.len() == 2 {
                                let output = args.args.first().expect("Expected output type");
                                let error = args.args.last().expect("Expected error type");
                                return (quote!(#output), quote!(#error));
                            }
                        }
                    }
                }
            }
            // If not a Result, wrap in Result with default error
            (quote!(#ty), quote!(::machi::tool::ToolError))
        }
        ReturnType::Default => (quote!(()), quote!(::machi::tool::ToolError)),
    }
}
