//! Abstractions for temporarily _rooting_ handles to managed JavaScript memory.

use std::mem;
use std::os::raw::c_void;
use std::marker::PhantomData;
use std::cell::Cell;
use neon_runtime;
use neon_runtime::raw;
use mem::Handle;
use js::{Value, JsObject};
use vm::internal::Isolate;
use self::internal::ScopeInternal;

pub(crate) mod internal {
    use vm::internal::Isolate;

    pub trait ScopeInternal: Sized {
        fn isolate(&self) -> Isolate;
        fn active(&self) -> bool;
        fn set_active(&self, bool);
    }
}

pub trait Scope<'a>: ScopeInternal {
    fn nested<T, F: for<'inner> FnOnce(&mut NestedScope<'inner>) -> T>(&self, f: F) -> T;
    fn chained<T, F: for<'inner> FnOnce(&mut ChainedScope<'inner, 'a>) -> T>(&self, f: F) -> T;

    fn global(&self) -> Handle<'a, JsObject> {
        JsObject::build(|out| {
            unsafe {
                neon_runtime::scope::get_global(self.isolate().to_raw(), out);
            }
        })
    }
}

#[inline]
fn ensure_active<T: ScopeInternal>(scope: &T) {
    if !scope.active() {
        panic!("illegal attempt to nest in inactive scope");
    }
}

pub struct RootScope<'a> {
    isolate: Isolate,
    active: Cell<bool>,
    phantom: PhantomData<&'a ()>
}

pub struct NestedScope<'a> {
    isolate: Isolate,
    active: Cell<bool>,
    phantom: PhantomData<&'a ()>
}

pub struct ChainedScope<'a, 'outer> {
    isolate: Isolate,
    active: Cell<bool>,
    v8: *mut raw::EscapableHandleScope,
    parent: PhantomData<&'outer ()>,
    phantom: PhantomData<&'a ()>
}

impl<'a, 'outer> ChainedScope<'a, 'outer> {
    #[inline]
    pub fn escape<T: Value>(&self, local: Handle<'a, T>) -> Handle<'outer, T> {
        unsafe {
            let mut result_local: raw::Local = mem::zeroed();
            neon_runtime::scope::escape(&mut result_local, self.v8, local.to_raw());
            Handle::new_internal(T::from_raw(result_local))
        }
    }
}

impl<'a> RootScope<'a> {
    #[inline]
    pub(crate) fn new(isolate: Isolate) -> RootScope<'a> {
        RootScope {
            isolate: isolate,
            active: Cell::new(true),
            phantom: PhantomData
        }
    }

    #[inline]
    pub(crate) fn with<T, F: FnOnce(&'a mut RootScope<'a>) -> T>(&'a mut self, f: F) -> T {
        debug_assert!(unsafe { neon_runtime::scope::size() } <= mem::size_of::<raw::HandleScope>());
        debug_assert!(unsafe { neon_runtime::scope::alignment() } <= mem::align_of::<raw::HandleScope>());

        #[cfg(feature = "v8_scope")]
        {
            let mut v8_scope = raw::HandleScope::new();

            unsafe {
                neon_runtime::scope::enter(&mut v8_scope, self.isolate().to_raw());
            }

            let result = f(self);

            unsafe {
                neon_runtime::scope::exit(&mut v8_scope);
            }

            result
        }

        #[cfg(not(feature = "v8_scope"))]
        f(self)
    }
}

impl<'a> Scope<'a> for RootScope<'a> {
    #[inline]
    fn nested<T, F: for<'inner> FnOnce(&mut NestedScope<'inner>) -> T>(&self, f: F) -> T {
        nest(self, f)
    }

    #[inline]
    fn chained<T, F: for<'inner> FnOnce(&mut ChainedScope<'inner, 'a>) -> T>(&self, f: F) -> T {
        chain(self, f)
    }
}

extern "C" fn chained_callback<'a, T, P, F>(out: &mut Box<Option<T>>,
                                            parent: &P,
                                            v8: *mut raw::EscapableHandleScope,
                                            f: Box<F>)
    where P: Scope<'a>,
          F: for<'inner> FnOnce(&mut ChainedScope<'inner, 'a>) -> T
{
    let mut chained = ChainedScope {
        isolate: parent.isolate(),
        active: Cell::new(true),
        v8: v8,
        parent: PhantomData,
        phantom: PhantomData
    };
    let result = f(&mut chained);
    **out = Some(result);
}

impl<'a> ScopeInternal for RootScope<'a> {
    #[inline]
    fn isolate(&self) -> Isolate { self.isolate }

    #[inline]
    fn active(&self) -> bool {
        self.active.get()
    }

    #[inline]
    fn set_active(&self, active: bool) {
        self.active.set(active);
    }
}

fn chain<'a, T, S, F>(outer: &S, f: F) -> T
    where S: Scope<'a>,
          F: for<'inner> FnOnce(&mut ChainedScope<'inner, 'a>) -> T
{
    ensure_active(outer);
    let closure: Box<F> = Box::new(f);
    let callback: extern "C" fn(&mut Box<Option<T>>, &S, *mut raw::EscapableHandleScope, Box<F>) = chained_callback::<T, S, F>;
    let mut result: Box<Option<T>> = Box::new(None);
    {
        let out: &mut Box<Option<T>> = &mut result;
        outer.set_active(false);
        unsafe {
            let out: *mut c_void = mem::transmute(out);
            let closure: *mut c_void = mem::transmute(closure);
            let callback: extern "C" fn(&mut c_void, *mut c_void, *mut c_void, *mut c_void) = mem::transmute(callback);
            let this: *mut c_void = mem::transmute(outer);
            neon_runtime::scope::chained(out, closure, callback, this);
        }
        outer.set_active(true);
    }
    result.unwrap()
}

fn nest<'me, T, S, F>(outer: &'me S, f: F) -> T
    where S: ScopeInternal,
          F: for<'nested> FnOnce(&mut NestedScope<'nested>) -> T
{
    ensure_active(outer);
    let closure: Box<F> = Box::new(f);
    let callback: extern "C" fn(&mut Box<Option<T>>, Isolate, Box<F>) = nested_callback::<T, F>;
    let mut result: Box<Option<T>> = Box::new(None);
    {
        let out: &mut Box<Option<T>> = &mut result;
        outer.set_active(false);
        unsafe {
            let out: *mut c_void = mem::transmute(out);
            let closure: *mut c_void = mem::transmute(closure);
            let callback: extern "C" fn(&mut c_void, *mut c_void, *mut c_void) = mem::transmute(callback);
            let isolate: *mut c_void = mem::transmute(outer.isolate());
            neon_runtime::scope::nested(out, closure, callback, isolate);
        }
        outer.set_active(true);
    }
    result.unwrap()
}

extern "C" fn nested_callback<T, F>(out: &mut Box<Option<T>>,
                                    isolate: Isolate,
                                    f: Box<F>)
    where F: for<'nested> FnOnce(&mut NestedScope<'nested>) -> T
{
    let mut nested = NestedScope {
        isolate: isolate,
        active: Cell::new(true),
        phantom: PhantomData
    };
    let result = f(&mut nested);
    **out = Some(result);
}

impl<'a> Scope<'a> for NestedScope<'a> {
    #[inline]
    fn nested<T, F: for<'inner> FnOnce(&mut NestedScope<'inner>) -> T>(&self, f: F) -> T {
        nest(self, f)
    }

    #[inline]
    fn chained<T, F: for<'inner> FnOnce(&mut ChainedScope<'inner, 'a>) -> T>(&self, f: F) -> T {
        chain(self, f)
    }
}

impl<'a> ScopeInternal for NestedScope<'a> {
    #[inline]
    fn isolate(&self) -> Isolate { self.isolate }

    #[inline]
    fn active(&self) -> bool {
        self.active.get()
    }

    #[inline]
    fn set_active(&self, active: bool) {
        self.active.set(active);
    }
}

impl<'a, 'outer> Scope<'a> for ChainedScope<'a, 'outer> {
    #[inline]
    fn nested<T, F: for<'inner> FnOnce(&mut NestedScope<'inner>) -> T>(&self, f: F) -> T {
        nest(self, f)
    }

    #[inline]
    fn chained<T, F: for<'inner> FnOnce(&mut ChainedScope<'inner, 'a>) -> T>(&self, f: F) -> T {
        chain(self, f)
    }
}

impl<'a, 'outer> ScopeInternal for ChainedScope<'a, 'outer> {
    #[inline]
    fn isolate(&self) -> Isolate { self.isolate }

    #[inline]
    fn active(&self) -> bool {
        self.active.get()
    }

    #[inline]
    fn set_active(&self, active: bool) {
        self.active.set(active);
    }
}
