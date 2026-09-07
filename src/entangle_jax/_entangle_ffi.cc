#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "_entangle_ffi.h"

namespace ffi = xla::ffi;

namespace {

ffi::Error entangle_impl(const ffi::AnyBuffer payload, const ffi::AnyBuffer witness,
                          ffi::Result<ffi::AnyBuffer> out) {
  (void)witness;
  (void)payload;
  (void)out;
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    entangle_handler, entangle_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

}  // namespace

extern "C" void* entangle_handler_address() {
  return reinterpret_cast<void*>(entangle_handler);
}
