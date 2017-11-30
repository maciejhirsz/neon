//! Abstractions representing the JavaScript virtual machine and its control flow.

use std::mem;
use std::any::TypeId;
use std::error::Error;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::marker::PhantomData;
use std::collections::HashMap;
use std::os::raw::c_void;
use std::panic::UnwindSafe;
use neon_runtime;
use neon_runtime::raw;
use scope::{Scope, RootScope};
use js::{JsValue, Value, Object, JsObject, JsFunction};
use js::class::internal::ClassMetadata;
use js::error::{JsError, Kind};
use mem::{Handle, Managed};
use self::internal::LockState;

pub(crate) mod internal {
    use std::mem;
    use std::collections::HashSet;
    use std::os::raw::c_void;
    use cslice::CMutSlice;
    use neon_runtime;
    use neon_runtime::raw;
    use super::ClassMap;

    pub struct LockState {
        buffers: HashSet<usize>
    }

    impl LockState {
        #[inline]
        pub fn new() -> LockState {
            LockState { buffers: HashSet::new() }
        }

        #[inline]
        pub fn use_buffer(&mut self, buf: CMutSlice<u8>) {
            let p = buf.as_ptr() as usize;
            if !self.buffers.insert(p) {
                panic!("attempt to lock heap with duplicate buffers (0x{:x})", p);
            }
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct Isolate(*mut raw::Isolate);

    #[inline]
    extern "C" fn drop_class_map(map: Box<ClassMap>) {
        mem::drop(map);
    }

    impl Isolate {
        #[inline]
        pub(crate) fn to_raw(self) -> *mut raw::Isolate {
            let Isolate(ptr) = self;
            ptr
        }

        #[inline]
        pub(crate) fn class_map(&mut self) -> &mut ClassMap {
            let mut ptr: *mut c_void = unsafe { neon_runtime::class::get_class_map(self.to_raw()) };
            if ptr.is_null() {
                let b: Box<ClassMap> = Box::new(ClassMap::new());
                let raw = Box::into_raw(b);
                ptr = unsafe { mem::transmute(raw) };
                let free_map: *mut c_void = unsafe { mem::transmute(drop_class_map as usize) };
                unsafe {
                    neon_runtime::class::set_class_map(self.to_raw(), ptr, free_map);
                }
            }
            unsafe { mem::transmute(ptr) }
        }

        #[inline]
        pub(crate) fn current() -> Isolate {
            unsafe {
                mem::transmute(neon_runtime::call::current_isolate())
            }
        }
    }
}

#[derive(Debug)]
pub struct Throw;

impl Display for Throw {
    fn fmt(&self, fmt: &mut Formatter) -> FmtResult {
        fmt.write_str("JavaScript Error")
    }
}

impl Error for Throw {
    fn description(&self) -> &str {
        "javascript error"
    }
}

pub type VmResult<T> = Result<T, Throw>;
pub type JsResult<'b, T> = VmResult<Handle<'b, T>>;

pub(crate) struct ClassMap {
    map: HashMap<TypeId, ClassMetadata>
}

impl ClassMap {
    #[inline]
    fn new() -> ClassMap {
        ClassMap {
            map: HashMap::new()
        }
    }

    #[inline]
    pub fn get(&self, key: &TypeId) -> Option<&ClassMetadata> {
        self.map.get(key)
    }

    #[inline]
    pub fn set(&mut self, key: TypeId, val: ClassMetadata) {
        self.map.insert(key, val);
    }
}

#[repr(C)]
pub(crate) struct CallbackInfo {
    info: raw::FunctionCallbackInfo
}

impl CallbackInfo {
    #[inline]
    pub fn data<'a>(&self) -> Handle<'a, JsValue> {
        unsafe {
            let mut local: raw::Local = mem::zeroed();
            neon_runtime::call::data(&self.info, &mut local);
            Handle::new_internal(JsValue::from_raw(local))
        }
    }

    #[inline]
    pub fn scope(&self) -> RootScope {
        RootScope::new(unsafe {
            mem::transmute(neon_runtime::call::get_isolate(mem::transmute(self)))
        })
    }

    #[inline]
    pub fn set_return<'a, 'b, T: Value>(&'a self, value: Handle<'b, T>) {
        unsafe {
            neon_runtime::call::set_return(&self.info, value.to_raw())
        }
    }

    #[inline]
    pub fn as_call<'a, T: This>(&'a self, scope: &'a mut RootScope<'a>) -> FunctionCall<'a, T> {
        FunctionCall {
            info: self,
            scope: scope,
            arguments: Arguments {
                info: &self,
                phantom: PhantomData
            }
        }
    }

    #[inline]
    fn kind(&self) -> CallKind {
        if unsafe { neon_runtime::call::is_construct(mem::transmute(self)) } {
            CallKind::Construct
        } else {
            CallKind::Call
        }
    }

    #[inline]
    pub fn len(&self) -> i32 {
        unsafe {
            neon_runtime::call::len(&self.info)
        }
    }

    #[inline]
    fn defined(&self, i: i32) -> bool {
        // Cast to u32 so we don't have to check negative values
        (i as u32) < (self.len() as u32)
    }

    #[inline]
    pub fn get<'b, T: Scope<'b>>(&self, _: &mut T, i: i32) -> Option<Handle<'b, JsValue>> {
        if !self.defined(i) {
            return None;
        }
        unsafe {
            let mut local: raw::Local = mem::zeroed();
            neon_runtime::call::get(&self.info, i, &mut local);
            Some(Handle::new_internal(JsValue::from_raw(local)))
        }
    }

    #[inline]
    pub fn require<'b, T: Scope<'b>>(&self, _: &mut T, i: i32) -> JsResult<'b, JsValue> {
        if !self.defined(i) {
            return JsError::throw(Kind::TypeError, "not enough arguments");
        }
        unsafe {
            let mut local: raw::Local = mem::zeroed();
            neon_runtime::call::get(&self.info, i, &mut local);
            Ok(Handle::new_internal(JsValue::from_raw(local)))
        }
    }

    #[inline]
    pub fn this<'b, T: Scope<'b>>(&self, _: &mut T) -> raw::Local {
        unsafe {
            let mut local: raw::Local = mem::zeroed();
            neon_runtime::call::this(mem::transmute(&self.info), &mut local);
            local
        }
    }

    #[inline]
    pub fn callee<'a, T: Scope<'a>>(&self, _: &mut T) -> Handle<'a, JsFunction> {
        unsafe {
            let mut local: raw::Local = mem::zeroed();
            neon_runtime::call::callee(mem::transmute(&self.info), &mut local);
            Handle::new_internal(JsFunction::from_raw(local))
        }
    }
}

pub struct Module<'a> {
    pub exports: Handle<'a, JsObject>,
    pub scope: &'a mut RootScope<'a>
}

impl<'a> Module<'a> {
    pub fn initialize(exports: Handle<JsObject>, init: fn(Module) -> VmResult<()>) {
        let mut scope = RootScope::new(unsafe { mem::transmute(neon_runtime::object::get_isolate(exports.to_raw())) });
        unsafe {
            let kernel: *mut c_void = mem::transmute(init);
            let callback: extern "C" fn(*mut c_void, *mut c_void, *mut c_void) = mem::transmute(module_callback as usize);
            let exports: raw::Local = exports.to_raw();
            let scope: *mut c_void = mem::transmute(&mut scope);
            neon_runtime::module::exec_kernel(kernel, callback, exports, scope);
        }
    }
}

impl<'a> Module<'a> {
    #[inline]
    pub fn export<T: Value>(&mut self, key: &str, f: fn(Call) -> JsResult<T>) -> VmResult<()> {
        let value = JsFunction::new(self.scope, f)?.upcast::<JsValue>();
        self.exports.set(key, value)?;
        Ok(())
    }
}

extern "C" fn module_callback<'a>(kernel: fn(Module) -> VmResult<()>, exports: Handle<'a, JsObject>, scope: &'a mut RootScope<'a>) {
    let _ = kernel(Module {
        exports: exports,
        scope: scope
    });
}

/// A type that may be the type of a function's `this` binding.
pub unsafe trait This: Managed {
    fn as_this(h: raw::Local) -> Self;
}

pub struct FunctionCall<'a, T: This> {
    info: &'a CallbackInfo,
    pub scope: &'a mut RootScope<'a>,
    pub arguments: Arguments<'a, T>
}

impl<'a, T: This> UnwindSafe for FunctionCall<'a, T> { }

pub type Call<'a> = FunctionCall<'a, JsObject>;

#[derive(Clone, Copy, Debug)]
pub enum CallKind {
    Construct,
    Call
}

impl<'a, T: This> FunctionCall<'a, T> {
    #[inline]
    pub fn kind(&self) -> CallKind { self.info.kind() }
}

#[repr(C)]
pub struct Arguments<'a, T> {
    info: &'a CallbackInfo,
    phantom: PhantomData<T>
}

impl<'a, T: This> Arguments<'a, T> {
    #[inline]
    pub fn len(&self) -> i32 { self.info.len() }

    #[inline]
    pub fn get<'b, U: Scope<'b>>(&self, scope: &mut U, i: i32) -> Option<Handle<'b, JsValue>> {
        self.info.get(scope, i)
    }

    #[inline]
    pub fn require<'b, U: Scope<'b>>(&self, scope: &mut U, i: i32) -> JsResult<'b, JsValue> {
        self.info.require(scope, i)
    }

    #[inline]
    pub fn this<'b, U: Scope<'b>>(&self, scope: &mut U) -> Handle<'b, T> {
        Handle::new_internal(T::as_this(self.info.this(scope)))
    }

    #[inline]
    pub fn callee<'b, U: Scope<'b>>(&self, scope: &mut U) -> Handle<'b, JsFunction> {
        self.info.callee(scope)
    }
}

/// A kernel of callable code exported to JS. A kernel function can be exported
/// to the Neon runtime as a raw pointer coupled with an `extern "C"` callback
/// function pointer.
pub(crate) trait Kernel<T: Clone + Copy + Sized>: Sized {

    /// The static callback function that can be passed to the Neon runtime to
    /// be called by V8 when the callable code is invoked. The Neon runtime
    /// ensures that the kernel will be provided as the extra data field,
    /// wrapped as a V8 External, in the `CallbackInfo` argument.
    extern "C" fn callback(info: &CallbackInfo) -> T;

    /// Extracts the kernel from the V8 External value pointed to by the given
    /// handle.
    unsafe fn from_wrapper(raw::Local) -> Self;

    /// Converts the kernel function to a raw void pointer.
    fn as_ptr(self) -> *mut c_void;

    /// Exports the kernel as a pair consisting of the static callback function
    /// and the kernel function, both converted to raw void pointers.
    #[inline]
    fn export(self) -> (*mut c_void, *mut c_void) {
        unsafe {
            (mem::transmute(Self::callback as usize), self.as_ptr())
        }
    }
}

pub trait Lock: Sized {
    type Internals;

    #[inline]
    fn grab<F, T>(self, f: F) -> T
        where F: FnOnce(Self::Internals) -> T + Send
    {
        let mut state = LockState::new();
        let internals = unsafe { self.expose(&mut state) };
        f(internals)
    }

    unsafe fn expose(self, state: &mut LockState) -> Self::Internals;
}

impl<T, U> Lock for (T, U)
    where T: Lock, U: Lock
{
    type Internals = (T::Internals, U::Internals);

    #[inline]
    unsafe fn expose(self, state: &mut LockState) -> Self::Internals {
        (self.0.expose(state), self.1.expose(state))
    }
}

impl<T> Lock for Vec<T>
    where T: Lock
{
    type Internals = Vec<T::Internals>;

    unsafe fn expose(self, state: &mut LockState) -> Self::Internals {
        self.into_iter()
            .map(|x| x.expose(state))
            .collect()
    }
}
