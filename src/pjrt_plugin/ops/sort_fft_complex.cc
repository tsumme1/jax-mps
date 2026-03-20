// Sort, FFT, and complex op handlers.

#include <mlx/fft.h>

#include <complex>

#include "pjrt_plugin/ops/handler_utils.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace jax_mps {

namespace {

// Handler for stablehlo.sort
bool HandleSort(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                ExecContext& ctx) {
    auto sortOp = mlir::dyn_cast<mlir::stablehlo::SortOp>(op);
    if (!sortOp) {
        MPS_LOG_ERROR("stablehlo.sort: failed to cast\n");
        return false;
    }

    int dimension = static_cast<int>(sortOp.getDimension());
    bool isStable = sortOp.getIsStable();
    (void)isStable;  // MLX sort is always stable

    // Analyze comparator to determine sort direction.
    // The comparator block has args (lhs0, rhs0, lhs1, rhs1, ...) where
    // lhs_i/rhs_i are the pair for input i. A "normal" ascending comparator
    // returns compare LT, %arg0, %arg1 (lhs < rhs). But optimization passes
    // may produce compare GT, %arg1, %arg0 (rhs > lhs) which is equivalent.
    // We check which block arguments the compare uses to handle this correctly.
    bool ascending = true;
    auto& comparator = sortOp.getComparator();
    if (!comparator.empty()) {
        auto& block = comparator.front();
        // Find the compare op that feeds the return value
        auto& returnOp = block.back();
        mlir::stablehlo::CompareOp cmpOp = nullptr;
        if (returnOp.getNumOperands() > 0) {
            // Trace back from return to find the compare
            mlir::Value result = returnOp.getOperand(0);
            if (auto* defOp = result.getDefiningOp()) {
                cmpOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(defOp);
                if (!cmpOp) {
                    // Try through select: select(eq, tiebreak, primary)
                    if (auto selOp = mlir::dyn_cast<mlir::stablehlo::SelectOp>(defOp)) {
                        if (auto* falseDefOp = selOp.getOnFalse().getDefiningOp()) {
                            cmpOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(falseDefOp);
                        }
                        if (!cmpOp) {
                            if (auto* trueDefOp = selOp.getOnTrue().getDefiningOp()) {
                                cmpOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(trueDefOp);
                            }
                        }
                    }
                    // Try through or: or(primary_lt, and(eq, tiebreak))
                    if (!cmpOp) {
                        if (auto orOp = mlir::dyn_cast<mlir::stablehlo::OrOp>(defOp)) {
                            if (auto* lhsDefOp = orOp.getLhs().getDefiningOp()) {
                                cmpOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(lhsDefOp);
                            }
                        }
                    }
                }
            }
        }
        // Fallback: scan for first CompareOp in the block
        if (!cmpOp) {
            for (auto& compOp : block.getOperations()) {
                if (auto found = mlir::dyn_cast<mlir::stablehlo::CompareOp>(compOp)) {
                    cmpOp = found;
                    break;
                }
            }
        }
        if (cmpOp) {
            auto dir = cmpOp.getComparisonDirection();
            bool isGtGe = (dir == mlir::stablehlo::ComparisonDirection::GT ||
                           dir == mlir::stablehlo::ComparisonDirection::GE);
            // Check if the compare operands are swapped relative to the canonical
            // order (arg0=lhs, arg1=rhs). If so, the effective direction is inverted.
            // E.g., GT(%arg1, %arg0) is equivalent to LT(%arg0, %arg1).
            bool swapped = false;
            if (auto lhsArg = mlir::dyn_cast<mlir::BlockArgument>(cmpOp.getLhs())) {
                if (auto rhsArg = mlir::dyn_cast<mlir::BlockArgument>(cmpOp.getRhs())) {
                    // Canonical: lhs uses even arg (0, 2, ...), rhs uses odd arg (1, 3, ...)
                    // Swapped: lhs uses odd arg, rhs uses even arg
                    swapped = (lhsArg.getArgNumber() % 2 == 1) && (rhsArg.getArgNumber() % 2 == 0);
                }
            }
            if (swapped) {
                ascending = isGtGe;  // GT(rhs, lhs) = LT(lhs, rhs) = ascending
            } else {
                ascending = !isGtGe;  // GT(lhs, rhs) = descending
            }
        }
    }

    size_t numInputs = sortOp.getInputs().size();

    if (numInputs == 1) {
        auto input_opt = GetValue(values, sortOp.getInputs()[0]);
        if (!input_opt) {
            MPS_LOG_ERROR("stablehlo.sort: input not found\n");
            return false;
        }
        // Sort descending by negating, sorting ascending, then negating back.
        if (ascending) {
            values.emplace(ToKey(op->getResult(0)), mlx::core::sort(input_opt->get(), dimension));
        } else {
            auto result = mlx::core::sort(mlx::core::negative(input_opt->get()), dimension);
            values.emplace(ToKey(op->getResult(0)), mlx::core::negative(result));
        }
    } else {
        // Sort-by-key
        auto keys_opt = GetValue(values, sortOp.getInputs()[0]);
        if (!keys_opt) {
            MPS_LOG_ERROR("stablehlo.sort: keys not found\n");
            return false;
        }

        // Sort descending by negating keys (argsort always sorts ascending).
        // This avoids the reverse-via-slice approach which has issues with
        // MLX's C++ slice API and negative stop values.
        auto sortKeys = ascending ? keys_opt->get() : mlx::core::negative(keys_opt->get());
        auto indices = mlx::core::argsort(sortKeys, dimension);

        for (size_t i = 0; i < numInputs; ++i) {
            auto input_opt = GetValue(values, sortOp.getInputs()[i]);
            if (!input_opt) {
                MPS_LOG_ERROR("stablehlo.sort: input %zu not found\n", i);
                return false;
            }
            auto sorted = mlx::core::take_along_axis(input_opt->get(), indices, dimension);
            values.emplace(ToKey(op->getResult(i)), std::move(sorted));
        }
    }

    return true;
}

// Handler for stablehlo.fft
bool HandleFft(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
               ExecContext& ctx) {
    auto fftOp = mlir::dyn_cast<mlir::stablehlo::FftOp>(op);
    if (!fftOp) {
        MPS_LOG_ERROR("stablehlo.fft: failed to cast\n");
        return false;
    }

    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.fft: operand not found\n");
        return false;
    }

    auto& input = input_opt->get();
    auto fftType = fftOp.getFftType();
    auto fftLength = fftOp.getFftLength();

    std::vector<int> axes;
    mlx::core::Shape lengths;
    int ndim = static_cast<int>(input.ndim());
    for (size_t i = 0; i < fftLength.size(); ++i) {
        axes.push_back(ndim - static_cast<int>(fftLength.size()) + static_cast<int>(i));
        lengths.push_back(static_cast<int>(fftLength[i]));
    }

    mlx::core::array result = input;
    switch (fftType) {
        case mlir::stablehlo::FftType::FFT:
            result = mlx::core::fft::fftn(input, lengths, axes);
            break;
        case mlir::stablehlo::FftType::IFFT:
            result = mlx::core::fft::ifftn(input, lengths, axes);
            break;
        case mlir::stablehlo::FftType::RFFT:
            result = mlx::core::fft::rfftn(input, lengths, axes);
            break;
        case mlir::stablehlo::FftType::IRFFT:
            result = mlx::core::fft::irfftn(input, lengths, axes);
            break;
        default:
            MPS_LOG_ERROR("stablehlo.fft: unsupported fft type\n");
            return false;
    }

    values.emplace(ToKey(op->getResult(0)), std::move(result));
    return true;
}

// Handler for stablehlo.complex (combine real + imag into complex)
bool HandleComplex(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                   ExecContext& ctx) {
    auto real_opt = GetValue(values, op->getOperand(0));
    auto imag_opt = GetValue(values, op->getOperand(1));
    if (!real_opt || !imag_opt) {
        MPS_LOG_ERROR("stablehlo.complex: operand not found\n");
        return false;
    }
    auto imag_unit = mlx::core::array(std::complex<float>(0.0F, 1.0F));
    auto result = mlx::core::add(
        mlx::core::astype(real_opt->get(), mlx::core::complex64),
        mlx::core::multiply(mlx::core::astype(imag_opt->get(), mlx::core::complex64), imag_unit));
    values.emplace(ToKey(op->getResult(0)), std::move(result));
    return true;
}

// Handler for chlo.top_k
bool HandleChloTopK(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                    ExecContext& ctx) {
    auto topKOp = mlir::dyn_cast<mlir::chlo::TopKOp>(op);
    if (!topKOp) {
        MPS_LOG_ERROR("chlo.top_k: failed to cast\n");
        return false;
    }

    auto input_opt = GetValue(values, topKOp.getOperand());
    if (!input_opt) {
        MPS_LOG_ERROR("chlo.top_k: operand not found\n");
        return false;
    }

    int k = static_cast<int>(topKOp.getK());
    int axis = static_cast<int>(input_opt->get().ndim()) - 1;

    auto input = mlx::core::contiguous(input_opt->get());
    auto sortedIndices = mlx::core::argsort(mlx::core::negative(input), axis);

    mlx::core::Shape starts(sortedIndices.ndim(), 0);
    mlx::core::Shape stops(sortedIndices.shape().begin(), sortedIndices.shape().end());
    stops[axis] = k;
    auto indices = mlx::core::slice(sortedIndices, starts, stops);

    auto topValues = mlx::core::take_along_axis(input, indices, axis);
    values.emplace(ToKey(op->getResult(0)), std::move(topValues));
    values.emplace(ToKey(op->getResult(1)), mlx::core::astype(indices, mlx::core::int32));
    return true;
}

}  // namespace

void RegisterSortFftComplexHandlers(std::unordered_map<std::string, OpHandler>& handlers) {
    handlers.insert({"stablehlo.sort", HandleSort});
    handlers.insert({"stablehlo.fft", HandleFft});
    handlers.insert({"stablehlo.complex", HandleComplex});
    handlers.insert({"chlo.top_k", HandleChloTopK});
}

}  // namespace jax_mps
