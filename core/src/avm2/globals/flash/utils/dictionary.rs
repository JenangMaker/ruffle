//! `flash.utils.Dictionary` natives

use crate::avm2::activation::Activation;
use crate::avm2::error::Error;
use crate::avm2::function::FunctionArgs;
use crate::avm2::value::Value;

pub use crate::avm2::object::dictionary_allocator;

/// `Dictionary.makeKeysWeak`, called by the constructor for `new Dictionary(true)`.
pub fn make_keys_weak<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: FunctionArgs<'_, 'gc>,
) -> Result<Value<'gc>, Error<'gc>> {
    if let Some(dictionary) = this.as_object().and_then(|o| o.as_dictionary_object()) {
        dictionary.set_weak_keys(activation);
    }
    Ok(Value::Undefined)
}
