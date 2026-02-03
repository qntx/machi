//! Tool macro implementation.
//!
//! This module provides the `#[tool]` attribute macro that transforms functions
//! into `Tool` trait implementations with automatic JSON schema generation.
//!
//! # Design Philosophy
//!
//! Inspired by smolagents' approach:
//! - Automatic metadata extraction from doc comments
//! - Zero-boilerplate tool definition
//! - Compile-time validation
//! - Explicit override capability when needed

use convert_case::{Case, Casing};
use darling::{FromMeta, ast::NestedMeta};
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use std::collections::HashMap;
use syn::{
    Attribute, Expr, ExprLit, FnArg, Ident, ItemFn, Lit, Meta, Pat, PathArguments, ReturnType,
    Type, parse_macro_input,
};

/// Unified JSON Schema type representation.
///
/// This enum provides a single source of truth for mapping Rust types to JSON Schema types,
/// eliminating code duplication between schema generation and output type inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JsonSchemaType {
    String,
    Integer,
    Number,
    Boolean,
    Array,
    Object,
    Null,
}

impl JsonSchemaType {
    /// Get the JSON Schema type string.
    const fn as_str(self) -> &'static str {
        match self {
            Self::String => "string",
            Self::Integer => "integer",
            Self::Number => "number",
            Self::Boolean => "boolean",
            Self::Array => "array",
            Self::Object => "object",
            Self::Null => "null",
        }
    }

    /// Infer JSON Schema type from a Rust type name.
    fn from_type_name(name: &str) -> Self {
        match name {
            // Integer types
            "i8" | "i16" | "i32" | "i64" | "i128" | "isize" | "u8" | "u16" | "u32" | "u64"
            | "u128" | "usize" => Self::Integer,
            // Floating point types
            "f32" | "f64" => Self::Number,
            // String types
            "String" | "str" | "Cow" => Self::String,
            // Boolean
            "bool" => Self::Boolean,
            // Array types
            "Vec" | "HashSet" | "BTreeSet" => Self::Array,
            // Null
            "()" => Self::Null,
            // Everything else is an object
            _ => Self::Object,
        }
    }
}

/// Information extracted from analyzing a Rust type.
#[derive(Debug, Clone)]
struct TypeInfo {
    /// The JSON Schema type
    schema_type: JsonSchemaType,
    /// Whether this type is nullable (Option<T>)
    nullable: bool,
    /// Inner type for generic types (e.g., T in Vec<T> or Option<T>)
    inner: Option<Box<TypeInfo>>,
}

impl TypeInfo {
    /// Analyze a Rust type and extract its JSON Schema information.
    fn from_type(ty: &Type) -> Self {
        match ty {
            Type::Path(type_path) => {
                let Some(segment) = type_path.path.segments.first() else {
                    return Self::object();
                };

                let type_name = segment.ident.to_string();

                // Handle Option<T>
                if type_name == "Option" {
                    if let Some(inner_ty) = Self::extract_generic_arg(segment) {
                        let mut inner_info = Self::from_type(inner_ty);
                        return Self {
                            schema_type: inner_info.schema_type,
                            nullable: true,
                            inner: inner_info.inner.take(),
                        };
                    }
                    return Self {
                        schema_type: JsonSchemaType::Object,
                        nullable: true,
                        inner: None,
                    };
                }

                // Handle Vec<T>, HashSet<T>, etc.
                if matches!(type_name.as_str(), "Vec" | "HashSet" | "BTreeSet") {
                    let inner =
                        Self::extract_generic_arg(segment).map(|ty| Box::new(Self::from_type(ty)));
                    return Self {
                        schema_type: JsonSchemaType::Array,
                        nullable: false,
                        inner,
                    };
                }

                // Handle HashMap<K, V>
                if matches!(type_name.as_str(), "HashMap" | "BTreeMap") {
                    return Self {
                        schema_type: JsonSchemaType::Object,
                        nullable: false,
                        inner: None,
                    };
                }

                // Handle Result<T, E> - extract the success type
                if type_name == "Result" {
                    if let Some(inner_ty) = Self::extract_generic_arg(segment) {
                        return Self::from_type(inner_ty);
                    }
                }

                Self {
                    schema_type: JsonSchemaType::from_type_name(&type_name),
                    nullable: false,
                    inner: None,
                }
            }
            Type::Reference(type_ref) => {
                // Handle &str specially
                if let Type::Path(inner) = &*type_ref.elem {
                    if inner.path.is_ident("str") {
                        return Self {
                            schema_type: JsonSchemaType::String,
                            nullable: false,
                            inner: None,
                        };
                    }
                }
                Self::from_type(&type_ref.elem)
            }
            Type::Tuple(tuple) if tuple.elems.is_empty() => Self {
                schema_type: JsonSchemaType::Null,
                nullable: false,
                inner: None,
            },
            _ => Self::object(),
        }
    }

    /// Create an object type info (fallback).
    fn object() -> Self {
        Self {
            schema_type: JsonSchemaType::Object,
            nullable: false,
            inner: None,
        }
    }

    /// Extract the first generic argument from a path segment.
    fn extract_generic_arg(segment: &syn::PathSegment) -> Option<&Type> {
        if let PathArguments::AngleBracketed(args) = &segment.arguments {
            if let Some(syn::GenericArgument::Type(ty)) = args.args.first() {
                return Some(ty);
            }
        }
        None
    }

    /// Generate the JSON Schema tokens for this type.
    fn to_schema_tokens(&self) -> TokenStream2 {
        let type_str = self.schema_type.as_str();

        match self.schema_type {
            JsonSchemaType::Array => {
                if let Some(inner) = &self.inner {
                    let inner_schema = inner.to_schema_tokens();
                    if self.nullable {
                        quote! { "type": #type_str, "items": { #inner_schema }, "nullable": true }
                    } else {
                        quote! { "type": #type_str, "items": { #inner_schema } }
                    }
                } else if self.nullable {
                    quote! { "type": #type_str, "nullable": true }
                } else {
                    quote! { "type": #type_str }
                }
            }
            _ => {
                if self.nullable {
                    quote! { "type": #type_str, "nullable": true }
                } else {
                    quote! { "type": #type_str }
                }
            }
        }
    }

    /// Get the output type string for LLM prompts.
    fn output_type_str(&self) -> &'static str {
        self.schema_type.as_str()
    }
}

/// State machine for parsing doc comment sections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum DocSection {
    #[default]
    Description,
    Arguments,
    Returns,
    Other,
}

/// Parsed documentation from doc comments.
#[derive(Debug, Default)]
struct DocInfo {
    description: String,
    param_descriptions: HashMap<String, String>,
    returns: Option<String>,
}

impl DocInfo {
    /// Parse doc comments from function attributes.
    fn from_attrs(attrs: &[Attribute]) -> Self {
        let doc_lines: Vec<String> = attrs
            .iter()
            .filter(|attr| attr.path().is_ident("doc"))
            .filter_map(|attr| {
                if let Meta::NameValue(meta) = &attr.meta {
                    if let Expr::Lit(ExprLit {
                        lit: Lit::Str(s), ..
                    }) = &meta.value
                    {
                        return Some(s.value());
                    }
                }
                None
            })
            .collect();

        if doc_lines.is_empty() {
            return Self::default();
        }

        let mut info = Self::default();
        let mut section = DocSection::default();
        let mut current_param: Option<String> = None;
        let mut description_lines = Vec::new();

        for line in &doc_lines {
            let trimmed = line.trim();

            // Detect section headers
            if let Some(new_section) = Self::detect_section(trimmed) {
                section = new_section;
                current_param = None;
                continue;
            }

            match section {
                DocSection::Description => {
                    description_lines.push(trimmed.to_string());
                }
                DocSection::Arguments => {
                    if let Some((name, desc)) = Self::parse_param_line(trimmed) {
                        info.param_descriptions.insert(name.clone(), desc);
                        current_param = Some(name);
                    } else if let Some(ref param) = current_param {
                        // Continuation line
                        if !trimmed.is_empty() {
                            if let Some(desc) = info.param_descriptions.get_mut(param) {
                                desc.push(' ');
                                desc.push_str(trimmed);
                            }
                        }
                    }
                }
                DocSection::Returns => {
                    if let Some(ref mut ret) = info.returns {
                        if !trimmed.is_empty() {
                            ret.push(' ');
                            ret.push_str(trimmed);
                        }
                    } else {
                        info.returns = Some(trimmed.to_string());
                    }
                }
                DocSection::Other => {}
            }
        }

        info.description = description_lines
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" ");

        info
    }

    /// Detect section header from a line.
    fn detect_section(line: &str) -> Option<DocSection> {
        if line.starts_with("# Arguments") || line.starts_with("# Args") {
            Some(DocSection::Arguments)
        } else if line.starts_with("# Returns") || line.starts_with("# Return") {
            Some(DocSection::Returns)
        } else if line.starts_with("# ") {
            Some(DocSection::Other)
        } else {
            None
        }
    }

    /// Parse a parameter documentation line: `* `param_name` - Description`
    fn parse_param_line(line: &str) -> Option<(String, String)> {
        let rest = line.strip_prefix("* `")?;
        let end_pos = rest.find('`')?;
        let param_name = rest[..end_pos].to_string();
        let desc = rest[end_pos + 1..]
            .trim_start_matches(" - ")
            .trim()
            .to_string();
        Some((param_name, desc))
    }
}

/// Parsed macro arguments for the `#[tool]` attribute.
///
/// Uses `darling` for declarative attribute parsing following community best practices.
#[derive(Debug, Default, FromMeta)]
#[darling(default)]
struct ToolArgs {
    /// Tool description (overrides doc comment)
    description: Option<String>,
    /// Parameter descriptions as key-value pairs
    #[darling(default)]
    params: ParamDescriptions,
    /// List of required parameter names
    #[darling(default, multiple)]
    required: Vec<String>,
    /// Output type hint for LLM prompts
    output_type: Option<String>,
    /// JSON schema for structured output (as JSON string)
    output_schema: Option<String>,
}

/// Wrapper for parameter descriptions to implement FromMeta.
#[derive(Debug, Default)]
struct ParamDescriptions(HashMap<String, String>);

impl FromMeta for ParamDescriptions {
    fn from_list(items: &[NestedMeta]) -> darling::Result<Self> {
        let mut map = HashMap::new();
        for item in items {
            if let NestedMeta::Meta(Meta::NameValue(nv)) = item {
                if let Some(ident) = nv.path.get_ident() {
                    if let Expr::Lit(ExprLit {
                        lit: Lit::Str(s), ..
                    }) = &nv.value
                    {
                        map.insert(ident.to_string(), s.value());
                    }
                }
            }
        }
        Ok(Self(map))
    }
}

impl ToolArgs {
    /// Parse from attribute arguments.
    fn from_args(args: TokenStream) -> darling::Result<Self> {
        let attr_args = NestedMeta::parse_meta_list(args.into())?;
        Self::from_list(&attr_args)
    }

    /// Get parameter descriptions map.
    fn param_descriptions(&self) -> &HashMap<String, String> {
        &self.params.0
    }
}

/// Rust keywords that cannot be used as parameter names.
const RUST_KEYWORDS: &[&str] = &[
    "as", "break", "const", "continue", "crate", "else", "enum", "extern", "false", "fn", "for",
    "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub", "ref", "return",
    "self", "Self", "static", "struct", "super", "trait", "true", "type", "unsafe", "use", "where",
    "while", "async", "await", "dyn",
];

/// Validate tool function at compile time.
fn validate_tool_function(input_fn: &ItemFn) -> syn::Result<()> {
    let fn_name = &input_fn.sig.ident;

    // Check return type is Result<T, E>
    match &input_fn.sig.output {
        ReturnType::Default => {
            return Err(syn::Error::new_spanned(
                fn_name,
                "tool function must return Result<T, ToolError>",
            ));
        }
        ReturnType::Type(_, ty) => {
            if let Type::Path(type_path) = &**ty {
                if let Some(segment) = type_path.path.segments.last() {
                    if segment.ident != "Result" {
                        return Err(syn::Error::new_spanned(
                            ty,
                            "tool function must return Result<T, E>",
                        ));
                    }
                }
            }
        }
    }

    // Validate parameter names
    for arg in &input_fn.sig.inputs {
        if let FnArg::Typed(pat_type) = arg {
            if let Pat::Ident(ident) = &*pat_type.pat {
                let name = ident.ident.to_string();
                if RUST_KEYWORDS.contains(&name.as_str()) {
                    return Err(syn::Error::new_spanned(
                        &ident.ident,
                        format!("parameter name '{name}' is a Rust keyword"),
                    ));
                }
            }
        }
    }

    Ok(())
}

/// Extract Output and Error types from Result<T, E>.
fn extract_result_types(return_type: &ReturnType) -> (TokenStream2, TokenStream2) {
    if let ReturnType::Type(_, ty) = return_type {
        if let Type::Path(type_path) = &**ty {
            if let Some(segment) = type_path.path.segments.last() {
                if segment.ident == "Result" {
                    if let PathArguments::AngleBracketed(args) = &segment.arguments {
                        if args.args.len() == 2 {
                            let output = args.args.first().expect("output type");
                            let error = args.args.last().expect("error type");
                            return (quote!(#output), quote!(#error));
                        }
                    }
                }
            }
        }
        return (quote!(#ty), quote!(::machi::tool::ToolError));
    }
    (quote!(()), quote!(::machi::tool::ToolError))
}

/// Parameter information extracted from function signature.
struct ParamInfo {
    name: Ident,
    ty: Type,
    description: String,
    is_required: bool,
    type_info: TypeInfo,
}

impl ParamInfo {
    fn from_fn_arg(arg: &FnArg, macro_args: &ToolArgs, doc_info: &DocInfo) -> Option<Self> {
        let FnArg::Typed(pat_type) = arg else {
            return None;
        };
        let Pat::Ident(pat_ident) = &*pat_type.pat else {
            return None;
        };

        let name = pat_ident.ident.clone();
        let name_str = name.to_string();
        let ty = (*pat_type.ty).clone();
        let type_info = TypeInfo::from_type(&ty);

        let description = macro_args
            .param_descriptions()
            .get(&name_str)
            .cloned()
            .or_else(|| doc_info.param_descriptions.get(&name_str).cloned())
            .unwrap_or_else(|| format!("Parameter {name_str}"));

        let is_required = !type_info.nullable || macro_args.required.contains(&name_str);

        Some(Self {
            name,
            ty,
            description,
            is_required,
            type_info,
        })
    }
}

/// Main implementation of the tool macro.
pub fn tool_impl(args: TokenStream, input: TokenStream) -> TokenStream {
    // Parse macro arguments using darling
    let macro_args = match ToolArgs::from_args(args) {
        Ok(args) => args,
        Err(err) => return TokenStream::from(err.write_errors()),
    };
    let input_fn = parse_macro_input!(input as ItemFn);

    // Validate at compile time
    if let Err(err) = validate_tool_function(&input_fn) {
        return err.to_compile_error().into();
    }

    // Extract function metadata
    let fn_name = &input_fn.sig.ident;
    let fn_name_str = fn_name.to_string();
    let is_async = input_fn.sig.asyncness.is_some();
    let fn_visibility = &input_fn.vis;

    // Parse documentation
    let doc_info = DocInfo::from_attrs(&input_fn.attrs);

    // Extract and analyze parameters
    let params: Vec<ParamInfo> = input_fn
        .sig
        .inputs
        .iter()
        .filter_map(|arg| ParamInfo::from_fn_arg(arg, &macro_args, &doc_info))
        .collect();

    // Extract return type info
    let return_type = &input_fn.sig.output;
    let (output_type, error_type) = extract_result_types(return_type);

    // Determine output type string
    let output_type_str = macro_args.output_type.clone().unwrap_or_else(|| {
        if let ReturnType::Type(_, ty) = return_type {
            TypeInfo::from_type(ty).output_type_str().to_string()
        } else {
            "null".to_string()
        }
    });

    // Generate output_schema implementation
    let output_schema_impl = if let Some(schema_str) = &macro_args.output_schema {
        quote! {
            ::std::option::Option::Some(
                ::serde_json::from_str(#schema_str)
                    .expect("Invalid JSON in output_schema attribute")
            )
        }
    } else {
        quote! { ::std::option::Option::None }
    };

    // Generate names
    let struct_name = format_ident!("{}", fn_name_str.to_case(Case::Pascal));
    let params_struct_name = format_ident!("{}Args", struct_name);
    let static_name = format_ident!("{}", fn_name_str.to_uppercase());

    // Generate description
    let tool_description = macro_args
        .description
        .as_ref()
        .map(|d| quote! { #d.to_string() })
        .unwrap_or_else(|| {
            if !doc_info.description.is_empty() {
                let desc = &doc_info.description;
                quote! { #desc.to_string() }
            } else {
                quote! { format!("Tool function: {}", #fn_name_str) }
            }
        });

    // Collect parameter data for code generation
    let param_names: Vec<_> = params.iter().map(|p| &p.name).collect();
    let param_types: Vec<_> = params.iter().map(|p| &p.ty).collect();
    let param_descriptions: Vec<_> = params.iter().map(|p| &p.description).collect();
    let json_schemas: Vec<_> = params
        .iter()
        .map(|p| p.type_info.to_schema_tokens())
        .collect();
    let required_params: Vec<_> = params
        .iter()
        .filter(|p| p.is_required)
        .map(|p| p.name.to_string())
        .collect();

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

    // Generate the complete implementation
    let expanded = quote! {
        #[derive(Debug, Clone, ::serde::Deserialize, ::serde::Serialize)]
        #fn_visibility struct #params_struct_name {
            #(
                #[doc = #param_descriptions]
                pub #param_names: #param_types,
            )*
        }

        #input_fn

        #[derive(Debug, Clone, Copy, Default)]
        #fn_visibility struct #struct_name;

        #[::async_trait::async_trait]
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
                                #json_schemas,
                                "description": #param_descriptions
                            }
                        ),*
                    },
                    "required": [#(#required_params),*]
                })
            }

            fn output_type(&self) -> &'static str {
                #output_type_str
            }

            fn output_schema(&self) -> ::std::option::Option<::serde_json::Value> {
                #output_schema_impl
            }

            #call_impl
        }

        #fn_visibility static #static_name: #struct_name = #struct_name;
    };

    TokenStream::from(expanded)
}
