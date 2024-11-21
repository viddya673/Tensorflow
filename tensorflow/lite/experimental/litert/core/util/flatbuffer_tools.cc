// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"

#include <memory>
#include <utility>

#include "tensorflow/compiler/mlir/lite/allocation.h"
#include "tensorflow/lite/experimental/litert/core/filesystem.h"

#ifndef NDEBUG
// Make flatbuffers verifier `assert` in debug mode.
#define FLATBUFFERS_DEBUG_VERIFICATION_FAILURE

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers  // IWYU pragma: keep
#endif

#include <cstddef>
#include <cstdint>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "flatbuffers/verifier.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/stderr_reporter.h"

namespace litert::internal {

using ::flatbuffers::Verifier;
using ::tflite::VerifyModelBuffer;

namespace {

Expected<uint32_t> FindMetadataInd(const TflModel& model,
                                   absl::string_view key) {
  tflite::MetadataT* fb_metadata = nullptr;
  for (auto& m : model.metadata) {
    if (m->name == key) {
      fb_metadata = m.get();
      break;
    }
  }
  if (fb_metadata == nullptr) {
    return Error(kLiteRtStatusErrorNotFound);
  }
  return fb_metadata->buffer;
}

}  // namespace

absl::string_view FbBufToStr(const uint8_t* fb_data, size_t size) {
  auto fb_buf_raw = reinterpret_cast<const char*>(fb_data);
  return absl::string_view(fb_buf_raw, size);
}

absl::string_view FbBufToStr(absl::Span<const uint8_t> fb_buf) {
  auto fb_buf_raw = reinterpret_cast<const char*>(fb_buf.data());
  const size_t fb_buf_size = fb_buf.size();
  return absl::string_view(fb_buf_raw, fb_buf_size);
}

absl::Span<char> FbBufToStr(absl::Span<uint8_t> fb_buf) {
  return absl::MakeSpan(reinterpret_cast<char*>(fb_buf.data()), fb_buf.size());
}

absl::Span<char> FbBufToStr(uint8_t* fb_data, size_t size) {
  return absl::MakeSpan(reinterpret_cast<char*>(fb_data), size);
}

bool VerifyFlatbuffer(absl::Span<const uint8_t> buf) {
  return VerifyFlatbuffer(buf.data(), buf.size());
}

bool VerifyFlatbuffer(const uint8_t* buf, size_t buf_size) {
  flatbuffers::Verifier::Options options;
#ifndef NDEBUG
  options.assert = true;
#endif
  flatbuffers::Verifier verifier(buf, buf_size, options);
  return VerifyModelBuffer(verifier);
}

Expected<MutableBufferRef<uint8_t>> GetMetadata(absl::string_view key,
                                                TflModel& model) {
  auto buffer_ind = FindMetadataInd(model, key);
  if (!buffer_ind) {
    // Metadata key already has value.
    return buffer_ind.Error();
  }
  auto& fb_vec = model.buffers.at(*buffer_ind)->data;
  return MutableBufferRef<uint8_t>(fb_vec.data(), fb_vec.size());
}

Expected<BufferRef<uint8_t>> GetMetadata(absl::string_view key,
                                         const TflModel& model) {
  auto metadata = GetMetadata(key, const_cast<TflModel&>(model));
  if (!metadata) {
    return metadata.Error();
  }
  return *metadata;
}

LiteRtStatus PushMetadata(absl::string_view key, TflModel& model,
                          BufferRef<uint8_t> metadata) {
  auto buffer_ind = FindMetadataInd(model, key);
  if (buffer_ind) {
    // Metadata key already has value.
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto& new_metadata =
      model.metadata.emplace_back(std::make_unique<tflite::MetadataT>());
  new_metadata->name.assign(key.data(), key.size());

  const auto new_m_buffer_ind = model.buffers.size();
  new_metadata->buffer = new_m_buffer_ind;

  auto& new_buffer = model.buffers.emplace_back(std::make_unique<TflBuffer>());
  new_buffer->data.assign(metadata.Data(), metadata.Data() + metadata.Size());

  return kLiteRtStatusOk;
}

Expected<MutableBufferRef<uint8_t>> GetTflBuffer(TflModel& tfl_model,
                                                 uint32_t buffer_ind) {
  if (buffer_ind >= tfl_model.buffers.size()) {
    return Error(kLiteRtStatusErrorIndexOOB);
  }
  auto& tfl_data = tfl_model.buffers.at(buffer_ind)->data;
  return MutableBufferRef<uint8_t>(tfl_data.data(), tfl_data.size());
}

Expected<BufferRef<uint8_t>> GetTflBuffer(const TflModel& tfl_model,
                                          uint32_t buffer_ind) {
  auto buffer = GetTflBuffer(const_cast<TflModel&>(tfl_model), buffer_ind);
  if (!buffer) {
    return buffer.Error();
  }
  return *buffer;
}

Expected<TflBufferPtr> TakeBuffer(TflModel& tfl_model, uint32_t buffer_ind) {
  if (buffer_ind >= tfl_model.buffers.size()) {
    return Error(kLiteRtStatusErrorIndexOOB);
  }
  return std::move(tfl_model.buffers.at(buffer_ind));
}

Expected<uint32_t> PushTflBuffer(TflModel& tfl_model,
                                 BufferRef<uint8_t> buffer) {
  tfl_model.buffers.emplace_back(std::make_unique<::tflite::BufferT>())
      ->data.assign(buffer.Data(), buffer.Data() + buffer.Size());
  return tfl_model.buffers.size() - 1;
}

Expected<TflOpCode> GetTflOpCode(const TflModel& tfl_model,
                                 uint32_t op_code_ind) {
  if (op_code_ind >= tfl_model.operator_codes.size()) {
    return Error(kLiteRtStatusErrorIndexOOB);
  }
  return std::move(tfl_model.operator_codes.at(op_code_ind)->builtin_code);
}

::tflite::Allocation::Ptr MakeAllocation(BufferRef<uint8_t> buf) {
  return std::make_unique<::tflite::MemoryAllocation>(
      buf.Data(), buf.Size(), ::tflite::DefaultErrorReporter());
}

Expected<FlatbufferWrapper::Ptr> FlatbufferWrapper::CreateFromBuffer(
    OwningBufferRef<uint8_t>&& buffer) {
  if (!VerifyFlatbuffer(buffer.Data(), buffer.Size())) {
    return Error(kLiteRtStatusErrorInvalidFlatbuffer);
  }

  auto alloc = MakeAllocation(buffer);

  if (alloc == nullptr) {
    return Error(kLiteRtStatusErrorFileIO);
  }

  auto fb_model = ::tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(alloc->base()), alloc->bytes());
  if (fb_model == nullptr) {
    return Error(kLiteRtStatusErrorFileIO);
  }

  return FlatbufferWrapper::Ptr(new FlatbufferWrapper(
      std::move(fb_model), std::move(alloc), std::move(buffer)));
}

Expected<FlatbufferWrapper::Ptr> FlatbufferWrapper::CreateFromBuffer(
    BufferRef<uint8_t> buffer) {
  return FlatbufferWrapper::CreateFromBuffer(
      OwningBufferRef<uint8_t>(buffer.Data(), buffer.Size()));
}

Expected<FlatbufferWrapper::Ptr> FlatbufferWrapper::CreateFromTflFile(
    absl::string_view path) {
  auto buf = LoadBinaryFile(path);
  if (!buf) {
    return buf.Error();
  }
  return FlatbufferWrapper::CreateFromBuffer(std::move(*buf));
}

}  // namespace litert::internal
