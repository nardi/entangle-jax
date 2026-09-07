#include <cstdint>
#include <cstring>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "_testing_ffi.h"

namespace ffi = xla::ffi;

static double g_slots[16] = {0};

ffi::Error write_impl(const ffi::Buffer<ffi::DataType::S32> id,
                       const ffi::Buffer<ffi::DataType::F32> value,
                       ffi::Result<ffi::Buffer<ffi::DataType::S32>> out_id) {
  int32_t slot = *id.typed_data();
  g_slots[slot] = *value.typed_data();
  *out_id->typed_data() = slot;
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    write_handler, write_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::S32>>());

ffi::Error read_impl(const ffi::Buffer<ffi::DataType::S32> id,
                      ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_value) {
  int32_t slot = *id.typed_data();
  *out_value->typed_data() = g_slots[slot];
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    read_handler, read_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>());

extern "C" {
void* write_handler_address() {
  return reinterpret_cast<void*>(write_handler);
}

void* read_handler_address() {
  return reinterpret_cast<void*>(read_handler);
}

void reset_slots() {
  std::memset(g_slots, 0, sizeof(g_slots));
}
}
