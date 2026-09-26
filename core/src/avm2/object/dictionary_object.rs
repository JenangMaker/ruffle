//! Object representation for `flash.utils.Dictionary`

use crate::avm2::Error;
use crate::avm2::activation::Activation;
use crate::avm2::dynamic_map::DynamicKey;
use crate::avm2::object::script_object::ScriptObjectData;
use crate::avm2::object::{ClassObject, Object, TObject};
use crate::avm2::value::Value;
use crate::string::AvmString;
use core::fmt;
use gc_arena::collect::Trace;
use gc_arena::{Collect, Finalization, Gc, GcWeak, Mutation};
use ruffle_common::utils::HasPrefixField;
use std::cell::Cell;

/// A class instance allocator that allocates Dictionary objects.
pub fn dictionary_allocator<'gc>(
    class: ClassObject<'gc>,
    activation: &mut Activation<'_, 'gc>,
) -> Result<Object<'gc>, Error<'gc>> {
    let base = ScriptObjectData::new(class);

    Ok(DictionaryObject(Gc::new(
        activation.gc(),
        DictionaryObjectData {
            base,
            weak_keys: Cell::new(false),
        },
    ))
    .into())
}

/// An object that allows associations between objects and values.
///
/// This is implemented by way of "object space", parallel to the property
/// space that ordinary properties live in. This space has no namespaces, and
/// keys are objects instead of strings.
#[derive(Clone, Collect, Copy)]
#[collect(no_drop)]
pub struct DictionaryObject<'gc>(pub Gc<'gc, DictionaryObjectData<'gc>>);

#[derive(Clone, Collect, Copy, Debug)]
#[collect(no_drop)]
pub struct DictionaryObjectWeak<'gc>(pub GcWeak<'gc, DictionaryObjectData<'gc>>);

impl fmt::Debug for DictionaryObject<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DictionaryObject")
            .field("ptr", &Gc::as_ptr(self.0))
            .finish()
    }
}

#[derive(Clone, HasPrefixField)]
#[repr(C, align(8))]
pub struct DictionaryObjectData<'gc> {
    /// Base script object
    base: ScriptObjectData<'gc>,

    /// `new Dictionary(true)`: object keys do not keep their entries alive.
    /// Set once, by the constructor, before any entry exists.
    weak_keys: Cell<bool>,
}

// SAFETY: traces everything the derived impl would, except that a weak-keyed
// dictionary leaves its object-keyed entries to
// `Avm2::finalize_weak_dictionaries`, which resurrects or removes each of
// them before anything is swept.
unsafe impl<'gc> Collect<'gc> for DictionaryObjectData<'gc> {
    fn trace<C: Trace<'gc>>(&self, cc: &mut C) {
        if self.weak_keys.get() {
            self.base.trace_with_weak_object_keys(cc);
        } else {
            cc.trace(&self.base);
        }
    }
}

impl<'gc> DictionaryObject<'gc> {
    /// Makes object keys weak, as `new Dictionary(true)` asks, and registers
    /// the dictionary for `Avm2::finalize_weak_dictionaries`.
    pub fn set_weak_keys(self, activation: &mut Activation<'_, 'gc>) {
        if !self.0.weak_keys.replace(true) {
            let mc = activation.gc();
            activation
                .avm2()
                .register_weak_dictionary(mc, DictionaryObjectWeak(Gc::downgrade(self.0)));
        }
    }

    /// First half of a finalization pass: resurrects object values whose key
    /// is still alive. Returns whether anything was resurrected, in which
    /// case marking has to resume and the pass be repeated.
    pub fn resurrect_values_of_live_keys(self, fc: &Finalization<'gc>) -> bool {
        let mut resurrected = false;
        let base = self.base();
        let values = base.values();
        for (key, property) in values.iter() {
            if let (DynamicKey::Object(key), Value::Object(value)) = (key, property.value) {
                let value = value.downgrade();
                if !key.downgrade().is_dead(fc) && value.is_dead(fc) {
                    value.resurrect(fc);
                    resurrected = true;
                }
            }
        }
        resurrected
    }

    /// Second half, once nothing more was resurrected: drops the entries whose
    /// key is about to be collected, before the sweep frees it.
    pub fn remove_dead_keys(self, fc: &Finalization<'gc>) {
        let dead: Vec<DynamicKey<'gc>> = self
            .base()
            .values()
            .iter()
            .filter_map(|(key, _)| match key {
                DynamicKey::Object(o) if o.downgrade().is_dead(fc) => Some(*key),
                _ => None,
            })
            .collect();
        if !dead.is_empty() {
            let base = self.base();
            let mut values = base.values_mut(fc);
            for key in &dead {
                values.remove(key);
            }
        }
    }

    /// Retrieve a value in the dictionary's object space.
    pub fn get_property_by_object(self, name: Object<'gc>) -> Value<'gc> {
        self.base()
            .values()
            .get(&DynamicKey::Object(name))
            .map(|v| v.value)
            .unwrap_or(Value::Undefined)
    }

    /// Set a value in the dictionary's object space.
    pub fn set_property_by_object(self, name: Object<'gc>, value: Value<'gc>, mc: &Mutation<'gc>) {
        self.base()
            .values_mut(mc)
            .insert(DynamicKey::Object(name), value);
    }

    /// Delete a value from the dictionary's object space.
    pub fn delete_property_by_object(self, name: Object<'gc>, mc: &Mutation<'gc>) {
        self.base().values_mut(mc).remove(&DynamicKey::Object(name));
    }

    pub fn has_property_by_object(self, name: Object<'gc>) -> bool {
        self.base().values().contains_key(&DynamicKey::Object(name))
    }
}

impl<'gc> TObject<'gc> for DictionaryObject<'gc> {
    fn gc_base(&self) -> Gc<'gc, ScriptObjectData<'gc>> {
        HasPrefixField::as_prefix_gc(self.0)
    }

    // Calling `setPropertyIsEnumerable` on a `Dictionary` has no effect -
    // stringified properties are always enumerable.
    fn set_local_property_is_enumerable(
        &self,
        _mc: &Mutation<'gc>,
        _name: AvmString<'gc>,
        _is_enumerable: bool,
    ) {
    }

    fn get_enumerant_value(
        self,
        index: u32,
        _activation: &mut Activation<'_, 'gc>,
    ) -> Result<Value<'gc>, Error<'gc>> {
        Ok(*self
            .base()
            .values()
            .value_at(index as usize)
            .unwrap_or(&Value::Undefined))
    }
}
