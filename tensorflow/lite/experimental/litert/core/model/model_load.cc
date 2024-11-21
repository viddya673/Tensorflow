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

#include "tensorflow/lite/experimental/litert/core/model/model_load.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/model/model_util.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace litert::internal {
namespace {

using GetBuffer = std::function<Expected<TflBufferPtr>(uint32_t ind)>;
using GetOpCode = std::function<Expected<LiteRtOpCode>(uint32_t ind)>;
using GetTensor = std::function<Expected<LiteRtTensorT::Ref>(size_t ind)>;

LiteRtRankedTensorType MapTensorType(const TflTensor& tfl_tensor) {
  LiteRtRankedTensorType ranked_tensor_type;
  ranked_tensor_type.element_type = MapElementType(tfl_tensor.type);
  ranked_tensor_type.layout.rank = tfl_tensor.shape.size();
  ranked_tensor_type.layout.dimensions = tfl_tensor.shape.data();
  // TFL tensors don't support strides yet.
  ranked_tensor_type.layout.strides = nullptr;
  return ranked_tensor_type;
}

LiteRtStatus ConvertTensor(const TflTensor& tfl_tensor, GetBuffer get_buffer,
                           LiteRtTensorT& target) {
  LITERT_RETURN_STATUS_IF_NOT_OK(IsTensorSupported(tfl_tensor));

  const auto buffer_ind = tfl_tensor.buffer;
  if (buffer_ind != 0) {
    auto buffer = get_buffer(tfl_tensor.buffer);
    if (!buffer) {
      return buffer.Error().Status();
    }
    LITERT_RETURN_STATUS_IF_NOT_OK(IsBufferSupported(**buffer));
    target.weights.fb_buffer = std::move(*buffer);
  }

  target.type_id = kLiteRtRankedTensorType;
  target.type_detail.ranked_tensor_type = MapTensorType(tfl_tensor);

  target.name = tfl_tensor.name;

  return kLiteRtStatusOk;
}

LiteRtStatus ConvertOp(const TflOp& op, GetTensor get_tensor,
                       GetOpCode get_op_code, LiteRtOpT& target) {
  auto op_code = get_op_code(op.opcode_index);
  if (!op_code) {
    return op_code.Error().Status();
  }
  target.op_code = *op_code;

  for (auto input_ind : op.inputs) {
    // Skipping optional input tensor.
    if (input_ind == -1) {
      continue;
    }

    auto input_tensor = get_tensor(input_ind);
    if (!input_tensor) {
      return input_tensor.Error().Status();
    }

    target.AddInput(input_tensor->get());
  }

  for (auto output_ind : op.outputs) {
    auto output_tensor = get_tensor(output_ind);
    if (!output_tensor) {
      return output_tensor.Error().Status();
    }

    target.AddOutput(output_tensor->get());
  }

  target.option = op.builtin_options;
  target.custom_options = OwningBufferRef<uint8_t>(op.custom_options.data(),
                                                   op.custom_options.size());

  return kLiteRtStatusOk;
}

class ModelUnpacker {
 public:
  static LiteRtStatus Unpack(LiteRtModel model);

 private:
  explicit ModelUnpacker(LiteRtModel model) : model_(model) {}

  LiteRtStatus UnpackSubgraph(LiteRtSubgraphT& target);

  GetBuffer GetBufferCallback() {
    return [&](auto buffer_ind) { return TakeBuffer(Fb(), buffer_ind); };
  }

  GetOpCode GetOpCodeCallback() {
    return [&](auto opcode_ind) -> Expected<LiteRtOpCode> {
      auto tfl_op_code = GetTflOpCode(Fb(), opcode_ind);
      if (!tfl_op_code) {
        return tfl_op_code.Error();
      }
      return static_cast<LiteRtOpCode>(*tfl_op_code);
    };
  }

  GetTensor GetTensorCallBack(const LiteRtSubgraphT& subgraph) {
    return [&](auto tensor_ind) -> Expected<LiteRtTensorT::Ref> {
      if (tensor_ind >= subgraph.tensors.size()) {
        return Error(kLiteRtStatusErrorIndexOOB);
      }
      return std::ref(*subgraph.tensors.at(tensor_ind));
    };
  }

  TflModel& Fb() { return *model_->flatbuffer_model; }

  LiteRtModel model_;
};

LiteRtStatus ModelUnpacker::UnpackSubgraph(LiteRtSubgraphT& target) {
  auto& flatbuffer_subgraph = target.flatbuffer_subgraph;

  for (auto& flatbuffer_tensor : flatbuffer_subgraph->tensors) {
    LITERT_RETURN_STATUS_IF_NOT_OK(IsTensorSupported(*flatbuffer_tensor));
    LITERT_RETURN_STATUS_IF_NOT_OK(ConvertTensor(
        *flatbuffer_tensor, GetBufferCallback(), target.EmplaceTensor()));
  }

  for (auto& flatbuffer_op : flatbuffer_subgraph->operators) {
    LITERT_RETURN_STATUS_IF_NOT_OK(
        ConvertOp(*flatbuffer_op, GetTensorCallBack(target),
                  GetOpCodeCallback(), target.EmplaceOp()));
  }

  for (auto input : flatbuffer_subgraph->inputs) {
    target.inputs.push_back(target.tensors[input]);
  }

  for (auto output : flatbuffer_subgraph->outputs) {
    target.outputs.push_back(target.tensors[output]);
  }

  return kLiteRtStatusOk;
}

LiteRtStatus ModelUnpacker::Unpack(LiteRtModel model) {
  ModelUnpacker unpacker(model);

  if (unpacker.Fb().subgraphs.size() != 1) {
    // TODO: b/365299994 - Support multi subgraph.
    LITERT_LOG(LITERT_ERROR, "%s",
               "Only models with 1 subgraph current supported\n");
    return kLiteRtStatusErrorUnsupported;
  }

  auto& subgraph = model->subgraphs.emplace_back();
  subgraph.flatbuffer_subgraph = std::move(unpacker.Fb().subgraphs[0]);
  LITERT_RETURN_STATUS_IF_NOT_OK(unpacker.UnpackSubgraph(subgraph));

  return kLiteRtStatusOk;
}

LiteRtStatus LoadModelFromFlatbuffer(std::unique_ptr<TflModel> flatbuffer,
                                     LiteRtModel* model) {
  auto litert_model = std::make_unique<LiteRtModelT>();
  litert_model->flatbuffer_model = std::move(flatbuffer);
  litert_model->subgraphs.reserve(100);

  LITERT_RETURN_STATUS_IF_NOT_OK(ModelUnpacker::Unpack(litert_model.get()));

  litert_model->flatbuffer_model->subgraphs.clear();

  *model = litert_model.release();

  return kLiteRtStatusOk;
}

}  // namespace

Expected<Model> LoadModelFromMemory(BufferRef<uint8_t> serialized) {
  auto flatbuffer = FlatbufferWrapper::CreateFromBuffer(serialized);
  if (!flatbuffer) {
    return flatbuffer.Error();
  }

  LiteRtModel model;
  LITERT_EXPECT_OK(LoadModelFromFlatbuffer(
      std::make_unique<TflModel>(std::move((*flatbuffer)->UnpackedModel())),
      &model));

  return Model::CreateFromOwnedHandle(model);
}

Expected<Model> LoadModelFromFile(absl::string_view path) {
  auto flatbuffer = FlatbufferWrapper::CreateFromTflFile(path);
  if (!flatbuffer) {
    return flatbuffer.Error();
  }

  LiteRtModel model;
  LITERT_EXPECT_OK(LoadModelFromFlatbuffer(
      std::make_unique<TflModel>(std::move((*flatbuffer)->UnpackedModel())),
      &model));

  return Model::CreateFromOwnedHandle(model);
}

}  // namespace litert::internal

LiteRtStatus LiteRtLoadModelFromMemory(const uint8_t* buf, size_t buf_size,
                                       LiteRtModel* model) {
  auto new_model = litert::internal::LoadModelFromMemory(
      litert::BufferRef<uint8_t>(buf, buf_size));
  if (!new_model) {
    return new_model.Error().Status();
  }
  *model = new_model->Release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtLoadModelFromFile(const char* path, LiteRtModel* model) {
  auto new_model = litert::internal::LoadModelFromFile(path);
  if (!new_model) {
    return new_model.Error().Status();
  }
  *model = new_model->Release();
  return kLiteRtStatusOk;
}
