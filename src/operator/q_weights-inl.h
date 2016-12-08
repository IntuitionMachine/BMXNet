/*!
 * Copyright (c) 2016 by Contributors
 * \file q_weights-inl.h
 * \brief Quantize Weights operator
 * \author HPI-DeepLearning
*/
#ifndef MXNET_OPERATOR_Q_WEIGHTS_INL_H_
#define MXNET_OPERATOR_Q_WEIGHTS_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include "./operator_common.h"
#include "./mshadow_op.h"

namespace mxnet {
    namespace op {
// Declare enumeration of input order to make code more intuitive.
// // These enums are only visible within this header
        namespace q_weights {
            enum QWeightsOpInputs {kData};
            enum QWeightsOpOutputs {kOut};
            enum QWeightsScalingFactor{kChannelMean, kScalar, kNone};
        }  // lowbit_weights

        struct QWeightsParam : public dmlc::Parameter<QWeightsParam> {
            unsigned int act_bit;
            int scaling_factor;
            DMLC_DECLARE_PARAMETER(QWeightsParam) {

                    DMLC_DECLARE_FIELD(act_bit).set_default(1).set_range(1, 32)
                            .describe("Number of bits weights should be quantized to (1-32)");

                    DMLC_DECLARE_FIELD(scaling_factor).set_default(q_weights::kNone)
                            .add_enum("channel_mean", q_weights::kChannelMean)
                            .add_enum("scalar", q_weights::kScalar)
                            .add_enum("none", q_weights::kNone)
                            .describe("Scaling factor to multiply binarized weight with.");
            }
        };

/**
 * \brief This is the implementation of quantizing weights operator.
 * \tparam xpu The device that the op will be executed on.
 */
        template<typename xpu, typename DType>
        class QWeightsOp : public Operator {
        public:
            explicit QWeightsOp(QWeightsParam param) {
                this->act_bit_ = param.act_bit;
                this->scaling_factor_ = param.scaling_factor;
            }
            virtual void Forward(const OpContext &ctx,
                                 const std::vector<TBlob> &in_data,
                                 const std::vector<OpReqType> &req,
                                 const std::vector<TBlob> &out_data,
                                 const std::vector<TBlob> &aux_args) {
                using namespace mshadow;
                using namespace mshadow::expr;
                CHECK_EQ(in_data.size(), 1);
                CHECK_EQ(out_data.size(), 1);
                Stream<xpu> *s = ctx.get_stream<xpu>();
                Tensor<xpu, 2, DType> data = in_data[q_weights::kData].FlatTo2D<xpu, DType>(s);
                Tensor<xpu, 2, DType> out = out_data[q_weights::kOut].FlatTo2D<xpu, DType>(s);

                if (act_bit_ == 32) {
                    Assign(out, req[q_weights::kOut], data);
                } else if (act_bit_ == 1) {
                    real_t scaling_factor = 1;
                    if (scaling_factor_ == q_weights::kScalar) {
                        scaling_factor = 5;
                    } else if (scaling_factor_ == q_weights::kChannelMean) {
                        LOG(FATAL) << "channel mean as scaling factor is currently not implemented";
                        scaling_factor = 0;
                    }
                    //Assign(out, req[q_weights::kOut], data / ScalarExp<DType>(scaling_factor));
                    Assign(out,
                           req[q_weights::kOut],
                           F<mshadow_op::det_sign>(data / ScalarExp<DType>(scaling_factor)) * ScalarExp<DType>(scaling_factor));
                } else {
                    LOG(FATAL) << "quantizing to n bits is currently not implemented (only 1 and 32 bit)";
//                    Assign(out, req[q_weights::kOut], F<mshadow_op::quantize>(
//                            F<mshadow_op::maximum>(F<mshadow_op::minimum>(data, scalar(DType(1))), scalar(DType(0))), //clip to [0, 1]
//                            scalar(DType(act_bit_))));
                }
            }

            virtual void Backward(const OpContext &ctx,
                                  const std::vector<TBlob> &out_grad,
                                  const std::vector<TBlob> &in_data,
                                  const std::vector<TBlob> &out_data,
                                  const std::vector<OpReqType> &req,
                                  const std::vector<TBlob> &in_grad,
                                  const std::vector<TBlob> &aux_args) {
                using namespace mshadow;
                using namespace mshadow::expr;
                CHECK_EQ(out_grad.size(), 1);
                CHECK(in_data.size() == 1 && in_grad.size() == 1);
                CHECK_EQ(req.size(), 1);
                Stream<xpu> *s = ctx.get_stream<xpu>();
                Tensor<xpu, 2, DType> m_out_grad = out_grad[q_weights::kOut].FlatTo2D<xpu, DType>(s);
                Tensor<xpu, 2, DType> m_in_data = in_data[q_weights::kData].FlatTo2D<xpu, DType>(s);
                Tensor<xpu, 2, DType> m_in_grad = in_grad[q_weights::kData].FlatTo2D<xpu, DType>(s);

                Assign(m_in_grad, req[q_weights::kData], F<mshadow_op::det_sign_grad>(m_in_data) * m_out_grad);

//                if (act_bit_ == 32) {
//                    Assign(m_in_grad, req[q_weights::kData], m_out_grad);
//                } else if (act_bit_ == 1) {
//                    Assign(m_in_grad, req[q_weights::kData], F<mshadow_op::det_sign_grad>(m_in_data) * m_out_grad);
//                } else {
//                    assert(false);
//                    Assign(m_in_grad, req[q_weights::kData], F<mshadow_op::quantize_grad>(m_in_data) * m_out_grad);
//                }
            }
        private:
            int act_bit_;
            int scaling_factor_;
        };  // class QWeightsOp

// Decalre Factory function, used for dispatch specialization
        template<typename xpu>
        Operator* CreateOp(QWeightsParam param, int dtype);

#if DMLC_USE_CXX11
        class QWeightsProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1) << "Input:[data]";
    const TShape &dshape = in_shape->at(q_weights::kData);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
          (*in_type)[i] = dtype;
      } else {
        CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                       << "Expected " << dtype << " v.s. given "
                                       << (*in_type)[i] << " at " << ListArguments()[i];
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new QWeightsProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "QWeights";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[q_weights::kOut], out_data[q_weights::kOut], in_data[q_weights::kData]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[q_weights::kOut], in_grad[q_weights::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[q_weights::kData], out_data[q_weights::kOut]}};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  QWeightsParam param_;
};
#endif  // DMLC_USE_CXX11
    }  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_Q_WEIGHTS_INL_H_
