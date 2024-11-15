// RUN: tf-opt -tfl-optimize-broadcasting -split-input-file %s | FileCheck %s

// CHECK-LABEL: FoldFillOpIntoDivOpRHS
func.func @FoldFillOpIntoDivOpRHS(%arg0: tensor<1x4x1440x1440xf32>, %arg1: tensor<4xi64>) -> tensor<1x4x1440x1440xf32> {
  %cst_2 = arith.constant dense<5.0> : tensor<f32>

  %1 = "tfl.fill"(%arg1, %cst_2) : (tensor<4xi64>, tensor<f32>) -> tensor<1x4x1440x1440xf32>
  %36 = tfl.div %arg0, %1 {fused_activation_function = "NONE"} : tensor<1x4x1440x1440xf32>
  return %36 : tensor<1x4x1440x1440xf32>
  // CHECK: %cst = arith.constant dense<5.000000e+00> : tensor<f32>
  // CHECK: %0 = tfl.div(%arg0, %cst) <{fused_activation_function = "NONE"}> : (tensor<1x4x1440x1440xf32>, tensor<f32>) -> tensor<1x4x1440x1440xf32>
  // CHECK: return %0 : tensor<1x4x1440x1440xf32>
}

// CHECK-LABEL: FoldFillOpIntoDivOpLHS
func.func @FoldFillOpIntoDivOpLHS(%arg0: tensor<1x4x1440x1440xf32>, %arg1: tensor<4xi64>) -> tensor<1x4x1440x1440xf32> {
  %cst_2 = arith.constant dense<5.0> : tensor<f32>

  %1 = "tfl.fill"(%arg1, %cst_2) : (tensor<4xi64>, tensor<f32>) -> tensor<1x4x1440x1440xf32>
  %36 = tfl.div %1, %arg0 {fused_activation_function = "NONE"} : tensor<1x4x1440x1440xf32>
  return %36 : tensor<1x4x1440x1440xf32>
  // CHECK: %cst = arith.constant dense<5.000000e+00> : tensor<f32>
  // CHECK: %0 = tfl.div(%cst, %arg0) <{fused_activation_function = "NONE"}> : (tensor<f32>, tensor<1x4x1440x1440xf32>) -> tensor<1x4x1440x1440xf32>
  // CHECK: return %0 : tensor<1x4x1440x1440xf32>
}

// CHECK-LABEL: FoldFillOpIntoMulOp
func.func @FoldFillOpIntoMulOp(%arg0: tensor<1x4x1440x1440xf32>, %arg1: tensor<4xi64>) -> tensor<1x4x1440x1440xf32> {
  %cst_2 = arith.constant dense<5.0> : tensor<f32>

  %1 = "tfl.fill"(%arg1, %cst_2) : (tensor<4xi64>, tensor<f32>) -> tensor<1x4x1440x1440xf32>
  %36 = tfl.mul %arg0, %1 {fused_activation_function = "NONE"} : tensor<1x4x1440x1440xf32>
  return %36 : tensor<1x4x1440x1440xf32>
  // CHECK: %cst = arith.constant dense<5.000000e+00> : tensor<f32>
  // CHECK: %0 = tfl.mul(%arg0, %cst) <{fused_activation_function = "NONE"}> : (tensor<1x4x1440x1440xf32>, tensor<f32>) -> tensor<1x4x1440x1440xf32>
  // CHECK: return %0 : tensor<1x4x1440x1440xf32>
}

//===----------------------------------------------------------------------===//
// Fuse splat constants into select_v2.
//===----------------------------------------------------------------------===//
// CHECK-LABEL: FuseSplatConstantsIntoSelectV2Rhs
func.func @FuseSplatConstantsIntoSelectV2Rhs(%arg0: tensor<1x8x128x1024xf32>, %arg1: tensor<1x1x128x1024xi1>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x8x128x1024xf32>
  %1 = "tfl.select_v2"(%arg1, %arg0, %cst_0) : (tensor<1x1x128x1024xi1>, tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = "tfl.select_v2"(%arg1, %arg0, %cst) : (tensor<1x1x128x1024xi1>, tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseSplatConstantsIntoSelectV2Lhs
func.func @FuseSplatConstantsIntoSelectV2Lhs(%arg0: tensor<1x8x128x1024xf32>, %arg1: tensor<1x1x128x1024xi1>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x8x128x1024xf32>
  %1 = "tfl.select_v2"(%arg1, %cst_0, %arg0) : (tensor<1x1x128x1024xi1>, tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = "tfl.select_v2"(%arg1, %cst, %arg0) : (tensor<1x1x128x1024xi1>, tensor<f32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

func.func @FuseSplatConstantsIntoSelectV2LhsInt8(%arg0: tensor<1x8x128x1024xi8>, %arg1: tensor<1x1x128x1024xi1>) -> (tensor<1x8x128x1024xi8>) {
  %cst_0 = arith.constant dense<0> : tensor<1x8x128x1024xi8>
  %1 = "tfl.select_v2"(%arg1, %cst_0, %arg0) : (tensor<1x1x128x1024xi1>, tensor<1x8x128x1024xi8>, tensor<1x8x128x1024xi8>) -> tensor<1x8x128x1024xi8>
  return %1 : tensor<1x8x128x1024xi8>
  // CHECK:  %cst = arith.constant dense<0> : tensor<i8>
  // CHECK:  %0 = "tfl.select_v2"(%arg1, %cst, %arg0) : (tensor<1x1x128x1024xi1>, tensor<i8>, tensor<1x8x128x1024xi8>) -> tensor<1x8x128x1024xi8>
  // CHECK:  return %0 : tensor<1x8x128x1024xi8>
}
