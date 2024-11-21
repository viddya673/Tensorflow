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

#include "tensorflow/lite/experimental/litert/core/model/model.h"

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAreArray;

TEST(ModelTest, GetMetadata) {
  LiteRtModelT model;
  model.flatbuffer_model = std::make_unique<tflite::ModelT>();

  static constexpr absl::string_view kMetadata = "VALUE";
  static constexpr absl::string_view kKey = "KEY";

  LITERT_ASSERT_STATUS_OK(
      model.PushMetadata(kKey, OwningBufferRef<uint8_t>(kMetadata)));
  auto found_metadata = model.FindMetadata(kKey);

  EXPECT_EQ(found_metadata->StrView(), kMetadata);
}

TEST(ModelTest, MetadataDNE) {
  LiteRtModelT model;
  model.flatbuffer_model = std::make_unique<tflite::ModelT>();

  auto res = model.FindMetadata("FOO");
  ASSERT_FALSE(res.HasValue());
}

TEST(ModelOpTest, AddInput) {
  LiteRtOpT op;
  LiteRtTensorT tensor;

  op.AddInput(tensor);

  EXPECT_THAT(tensor.users, ElementsAreArray({&op}));
  EXPECT_THAT(tensor.user_arg_inds, ElementsAreArray({0}));

  EXPECT_THAT(op.inputs, ElementsAreArray({&tensor}));
}

TEST(ModelOpTest, AddOutput) {
  LiteRtOpT op;
  LiteRtTensorT tensor;

  op.AddOutput(tensor);

  EXPECT_EQ(tensor.defining_op, &op);
  EXPECT_EQ(tensor.defining_op_out_ind, 0);

  EXPECT_THAT(op.outputs, ElementsAreArray({&tensor}));
}

TEST(ModelSubgraphTest, EmplaceTensor) {
  LiteRtSubgraphT subgraph;
  auto& tensor = subgraph.EmplaceTensor();
  ASSERT_EQ(subgraph.tensors_storage.size(), 1);
  EXPECT_THAT(subgraph.tensors, ElementsAreArray({&tensor}));
}

TEST(ModelSubgraphTest, EmplaceOp) {
  LiteRtSubgraphT subgraph;
  auto& tensor = subgraph.EmplaceOp();
  ASSERT_EQ(subgraph.ops_storage.size(), 1);
  EXPECT_THAT(subgraph.ops, ElementsAreArray({&tensor}));
}

}  // namespace
}  // namespace litert::internal
