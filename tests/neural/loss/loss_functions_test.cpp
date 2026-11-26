#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda/neural/loss/loss_functions.h>
#include "cuda_alloc_injection.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <random>
#include <vector>

namespace cuda::neural::loss::test {

    class LossFunctionsTest : public ::testing::Test {
    protected:
        void SetUp() override {
            cudaGetDevice(&device_);
            cudaStreamCreate(&stream_);
        }

        void TearDown() override { cudaStreamDestroy(stream_); }

        int device_ = 0;
        cudaStream_t stream_ = nullptr;
    };

    TEST_F(LossFunctionsTest, CrossEntropyLoss) {
        CrossEntropyConfig config;
        config.num_classes = 10;
        config.reduction_mean = true;

        int batch_size = 4;
        int num_classes = 10;

        std::vector<float> predictions(batch_size * num_classes);
        std::vector<int> targets(batch_size);
        std::vector<float> output(batch_size);

        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < num_classes; ++c) {
                predictions[b * num_classes + c] = (c == b % num_classes) ? 2.0f : 0.0f;
            }
            targets[b] = b % num_classes;
        }

        float loss = cross_entropy_loss(predictions.data(), targets.data(), output.data(), batch_size, num_classes, config, stream_);

        cudaStreamSynchronize(stream_);
        EXPECT_GT(loss, 0.0f);
        EXPECT_LT(loss, 10.0f);
    }

    TEST_F(LossFunctionsTest, FocalLoss) {
        FocalLossConfig config;
        config.num_classes = 5;
        config.alpha = 1.0f;
        config.gamma = 2.0f;

        int batch_size = 4;
        int num_classes = 5;

        std::vector<float> predictions(batch_size * num_classes);
        std::vector<int> targets(batch_size);
        std::vector<float> output(batch_size);

        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < num_classes; ++c) {
                predictions[b * num_classes + c] = 1.0f / num_classes;
            }
            targets[b] = 0;
        }

        float loss = focal_loss(predictions.data(), targets.data(), output.data(), batch_size, num_classes, config, stream_);

        cudaStreamSynchronize(stream_);
        EXPECT_GT(loss, 0.0f);
    }

    // focal_loss is the one entry in this file that validates nothing, while its four siblings all
    // do: cross_entropy_loss rejects batch_size < 0 and num_classes <= 0, returns 0.0f for an empty
    // batch, and range-checks every target; cross_entropy_logits_backward, cross_entropy_loss_device
    // and accuracy_device do the same. Three separate failures follow from the gap, all measured:
    //
    //   batch_size = -1  -> std::vector<float> probs(-1 * num_classes), a negative count converted
    //                       to size_t: length_error naming the container, not the parameter.
    //   batch_size = 0   -> `return total_loss / batch_size` divides by zero and returns NaN as
    //                       though it were a loss. cross_entropy_loss returns 0.0f for the same
    //                       input.
    //   target = 99, C=3 -> probs[b * num_classes + target_class] reads past the end of the vector
    //                       and the resulting loss is computed from whatever is there.
    TEST_F(LossFunctionsTest, FocalLossRejectsTheInputsItsSiblingsReject) {
        FocalLossConfig config;
        config.num_classes = 3;
        config.alpha = 1.0f;
        config.gamma = 2.0f;

        const int num_classes = 3;
        const int batch_size = 4;
        std::vector<float> predictions(static_cast<size_t>(batch_size) * num_classes, 1.0f);
        std::vector<int> targets = {0, 1, 2, 0};
        std::vector<float> output(batch_size);

        EXPECT_THROW((void)focal_loss(predictions.data(), targets.data(), output.data(), -1, num_classes, config, stream_), std::invalid_argument)
            << "a negative batch must be reported, not converted to a container size";
        EXPECT_THROW((void)focal_loss(predictions.data(), targets.data(), output.data(), batch_size, 0, config, stream_), std::invalid_argument)
            << "num_classes 0 has no softmax to take";

        // An empty batch is a zero loss, the way cross_entropy_loss reports it -- not 0/0.
        std::vector<float> empty_predictions, empty_output;
        std::vector<int> empty_targets;
        const float empty_loss = focal_loss(empty_predictions.data(), empty_targets.data(), empty_output.data(), 0, num_classes, config, stream_);
        EXPECT_FALSE(std::isnan(empty_loss)) << "an empty batch returned NaN";
        EXPECT_FLOAT_EQ(empty_loss, 0.0f);

        // A target outside [0, num_classes) must be refused, not read past the end of probs.
        std::vector<int> bad_targets = {0, 1, 99, 0};
        EXPECT_THROW((void)focal_loss(predictions.data(), bad_targets.data(), output.data(), batch_size, num_classes, config, stream_), std::invalid_argument)
            << "target 99 indexes past the end of a 12-element probability vector";
    }

    // Regression for the focal_loss softmax denominator bug (the same
    // class cross_entropy_loss was already fixed for): the old code divided by the
    // running partial sum, so probs[class0] was always 1.0 and only the last class
    // saw the correct denominator. Non-uniform logits + host fp64 reference of the
    // two-pass softmax must match element-for-element — RED against the broken
    // implementation.
    TEST_F(LossFunctionsTest, FocalLoss_SoftmaxReferenceParity) {
        FocalLossConfig config;
        config.num_classes = 5;
        config.alpha = 1.0f;
        config.gamma = 2.0f;

        int batch_size = 4;
        int num_classes = 5;

        // Distinct logits so the wrong partial-sum denominator is observable: for
        // row 0 the broken code would give probs[class0]==1.0 (loss ~0) regardless.
        std::vector<float> preds = {
            2.0f, 1.0f, 0.5f, -0.5f, -2.0f, 0.1f, 3.0f, 0.2f, -1.0f, 0.8f, -3.0f, -1.5f, 2.5f, 1.2f, 0.0f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f,
        };
        std::vector<int> targets = {0, 2, 3, 1};
        std::vector<float> output(batch_size);

        float mean = focal_loss(preds.data(), targets.data(), output.data(), batch_size, num_classes, config, stream_);
        (void)cudaGetLastError();  // focal_loss is host-side; drain sticky errors

        constexpr double eps = 1e-8;
        double ref_total = 0.0;
        for (int b = 0; b < batch_size; ++b) {
            const float* row = preds.data() + static_cast<size_t>(b) * num_classes;
            double max_logit = -INFINITY;
            for (int c = 0; c < num_classes; ++c) {
                max_logit = std::max(max_logit, static_cast<double>(row[c]));
            }
            double sum_exp = 0.0;
            for (int c = 0; c < num_classes; ++c) {
                sum_exp += std::exp(static_cast<double>(row[c]) - max_logit);
            }
            const int t = targets[b];
            double p = std::exp(static_cast<double>(row[t]) - max_logit) / sum_exp;
            p = std::max(std::min(p, 1.0 - eps), eps);
            double ref = -config.alpha * std::pow(1.0 - p, config.gamma) * std::log(p);
            SCOPED_TRACE("batch " + std::to_string(b));
            EXPECT_NEAR(output[b], static_cast<float>(ref), 1e-3f) << "per-element focal loss must match the two-pass softmax reference";
            ref_total += ref;
        }
        EXPECT_NEAR(mean, static_cast<float>(ref_total / batch_size), 1e-3f) << "mean focal loss must match the reference mean";
    }

    TEST_F(LossFunctionsTest, ContrastiveLoss) {
        ContrastiveLossConfig config;
        config.temperature = 0.1f;
        config.embedding_dim = 64;
        config.normalize_embeddings = true;

        int batch_size = 4;
        int embedding_dim = 64;

        std::vector<float> embeddings1(batch_size * embedding_dim, 0.1f);
        std::vector<float> embeddings2(batch_size * embedding_dim, 0.1f);
        std::vector<float> output(batch_size);

        float loss = contrastive_loss(embeddings1.data(), embeddings2.data(), output.data(), batch_size, embedding_dim, config, stream_);

        cudaStreamSynchronize(stream_);
        EXPECT_GE(loss, 0.0f);
    }

    // contrastive_loss guards its temperature but not the batch. With batch_size == 0 both loops are
    // skipped, sum_exp_pos and sum_exp_neg stay 0, and the final
    //
    //     float loss = -logf(sum_exp_pos / (sum_exp_pos + sum_exp_neg));    // 0/0
    //
    // returns NaN as though it were a loss. The two siblings in this file that also take a batch --
    // cross_entropy_loss and focal_loss -- both return 0.0f for an empty batch, which is the answer
    // the mean of no pairs has.
    TEST_F(LossFunctionsTest, ContrastiveLossReportsZeroForAnEmptyBatch) {
        ContrastiveLossConfig config;
        config.embedding_dim = 64;

        std::vector<float> empty_embeddings, empty_output;
        const float loss = contrastive_loss(empty_embeddings.data(), empty_embeddings.data(), empty_output.data(), 0, 64, config, stream_);

        EXPECT_FALSE(std::isnan(loss)) << "an empty batch returned NaN as a loss value";
        EXPECT_FLOAT_EQ(loss, 0.0f);

        // A negative batch has no meaning either; both siblings refuse it rather than constructing a
        // vector from it.
        EXPECT_THROW((void)contrastive_loss(empty_embeddings.data(), empty_embeddings.data(), empty_output.data(), -1, 64, config, stream_), std::invalid_argument);
    }

    TEST_F(LossFunctionsTest, CrossEntropyLossFunction) {
        CrossEntropyConfig config;
        config.num_classes = 3;

        CrossEntropyLossFunction loss_fn(config);
        EXPECT_EQ(loss_fn.get_type(), LossType::CrossEntropy);
        EXPECT_EQ(loss_fn.get_name(), "CrossEntropy");
    }

    TEST_F(LossFunctionsTest, FocalLossFunction) {
        FocalLossConfig config;
        config.num_classes = 3;

        FocalLossFunction loss_fn(config);
        EXPECT_EQ(loss_fn.get_type(), LossType::FocalLoss);
        EXPECT_EQ(loss_fn.get_name(), "FocalLoss");
    }

    TEST_F(LossFunctionsTest, ContrastiveLossFunction) {
        ContrastiveLossConfig config;
        config.embedding_dim = 64;

        ContrastiveLossFunction loss_fn(config);
        EXPECT_EQ(loss_fn.get_type(), LossType::ContrastiveLoss);
        EXPECT_EQ(loss_fn.get_name(), "ContrastiveLoss");
    }

    TEST_F(LossFunctionsTest, LossTypeEnum) {
        EXPECT_EQ(static_cast<int>(LossType::CrossEntropy), 0);
        EXPECT_EQ(static_cast<int>(LossType::FocalLoss), 1);
        EXPECT_EQ(static_cast<int>(LossType::ContrastiveLoss), 2);
    }

    TEST_F(LossFunctionsTest, CrossEntropyConfig) {
        CrossEntropyConfig config;
        config.num_classes = 10;
        config.label_smoothing = 0.1f;
        config.reduction_mean = true;

        EXPECT_EQ(config.num_classes, 10);
        EXPECT_EQ(config.label_smoothing, 0.1f);
        EXPECT_TRUE(config.reduction_mean);
    }

    // ============================================================================
    // Milestone v2.24 / RED: device cross-entropy-with-logits backward.
    // The host cross_entropy_loss() computes only the scalar (its device kernels
    // are dead code), so no grad-logits exist to seed a layer backward chain. This
    // test pins the analytic contract grad_logits[b*C+c] = (softmax_c - onehot)/B
    // (reduction_mean) against a host double-precision reference. RED against the
    // provisional throw-stub; GREEN with the P2 device kernel.
    // ============================================================================
    TEST_F(LossFunctionsTest, CrossEntropyLogitsBackwardDevice) {
        CrossEntropyConfig config;
        config.num_classes = 10;
        config.reduction_mean = true;

        const int batch = 6;
        const int C = 10;
        std::vector<float> logits(static_cast<size_t>(batch) * C);
        std::vector<int> targets(batch);
        std::mt19937 rng(20260813);
        std::normal_distribution<float> dist(0.0f, 2.0f);
        for (size_t i = 0; i < logits.size(); ++i) {
            logits[i] = dist(rng);
        }
        for (int b = 0; b < batch; ++b) {
            targets[b] = static_cast<int>(rng() % C);
        }

        // Host double-precision analytic grad: softmax(logits) - onehot, /batch.
        std::vector<float> ref(static_cast<size_t>(batch) * C);
        for (int b = 0; b < batch; ++b) {
            double m = logits[b * C];
            for (int c = 1; c < C; ++c) {
                m = std::max(m, static_cast<double>(logits[b * C + c]));
            }
            double sum = 0.0;
            for (int c = 0; c < C; ++c) {
                sum += std::exp(static_cast<double>(logits[b * C + c]) - m);
            }
            for (int c = 0; c < C; ++c) {
                double sm = std::exp(static_cast<double>(logits[b * C + c]) - m) / sum;
                double grad = sm - (c == targets[b] ? 1.0 : 0.0);
                ref[b * C + c] = static_cast<float>(grad / batch);
            }
        }

        cuda::memory::Buffer<float> d_logits(logits.size());
        cuda::memory::Buffer<int> d_targets(targets.size());
        cuda::memory::Buffer<float> d_grad(logits.size());
        d_logits.copy_from(logits.data(), logits.size());
        d_targets.copy_from(targets.data(), targets.size());

        std::vector<float> grad(logits.size(), 12345.0f);
        EXPECT_NO_THROW(cross_entropy_logits_backward(d_logits.data(), d_targets.data(), d_grad.data(), batch, C, config, stream_));
        d_grad.copy_to(grad.data(), grad.size());

        for (size_t i = 0; i < grad.size(); ++i) {
            EXPECT_NEAR(ref[i], grad[i], 1e-5f) << "grad_logits mismatch at " << i;
        }
    }

    // cross_entropy_loss_device takes scratch with cudaMallocAsync and releases it on its last
    // line, with four throwing calls in between:
    //
    //     CUDA_CHECK(cudaMallocAsync(&bad_target, sizeof(int), stream));
    //     CUDA_CHECK(cudaMemsetAsync(bad_target, 0, sizeof(int), stream));   // <- fail this
    //     ... kernel launch, cudaGetLastError, cudaMemcpyAsync, cudaStreamSynchronize ...
    //     CUDA_CHECK(cudaFreeAsync(bad_target, stream));    // reached only if all of the above succeed
    //
    // A failure in any of them unwinds without reaching the free, so the stream-ordered allocation
    // leaks. The business check (the invalid_argument for an offending target) sits after the free
    // and is fine; it is the CUDA error path that lacks the release.
    TEST_F(LossFunctionsTest, DeviceLossReleasesItsScratchWhenALaterCallFails) {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            (void)cudaGetLastError();
            GTEST_SKIP() << "no CUDA device";
        }
        (void)cudaGetLastError();

        const int batch = 4, C = 5;
        CrossEntropyConfig config;
        config.reduction_mean = true;

        std::vector<float> logits(static_cast<size_t>(batch) * C, 0.5f);
        std::vector<int> targets(batch, 0);
        cuda::memory::Buffer<float> d_logits(logits.size()), d_loss(1);
        cuda::memory::Buffer<int> d_targets(targets.size());
        d_logits.copy_from(logits.data(), logits.size());
        d_targets.copy_from(targets.data(), targets.size());

        cuda_test::reset_async_allocation_count();

        // The function memsets loss_out first, then takes the scratch and memsets that. Index 0
        // would fail the very first call, before any scratch exists, so let it through and fail
        // index 1 -- the memset of bad_target, which is the one after the allocation.
        cuda_test::fail_memset_after(1);
        EXPECT_THROW(cross_entropy_loss_device(d_logits.data(), d_targets.data(), d_loss.data(), batch, C, config, stream_), std::exception)
            << "the injected memset failure was expected to surface";
        cuda_test::stop_failing_memsets();

        EXPECT_EQ(cuda_test::live_async_allocation_count(), 0) << "the scratch must be released on the failure path, not only on the success one";
        (void)cudaGetLastError();
    }

    // ============================================================================
    // Milestone v2.30 — device cross-entropy-loss scalar.
    // cross_entropy_loss_device computes the identical mean CE loss as the host
    // cross_entropy_loss but from device logits/targets, writing a single scalar
    // to [1] so the training drivers no longer round-trip the full m*C logits
    // buffer through the host every step.
    // Pinned against a host double-precision reference. RED against the
    // provisional throw-stub; GREEN with the P2 device reduction kernel.
    // ============================================================================
    namespace {

        // Host double-precision reference for the mean CE loss (max-subtracted
        // softmax, target log-probability), matching the host cross_entropy_loss()
        // math exactly (the same full-class-sum ordering; no label smoothing).
        double host_ce_loss_reference(const std::vector<float>& logits, const std::vector<int>& targets, int batch, int C, bool reduction_mean) {
            double total = 0.0;
            for (int b = 0; b < batch; ++b) {
                double m = logits[static_cast<size_t>(b) * C];
                for (int c = 1; c < C; ++c) {
                    m = std::max(m, static_cast<double>(logits[static_cast<size_t>(b) * C + c]));
                }
                double sum_exp = 0.0;
                for (int c = 0; c < C; ++c) {
                    sum_exp += std::exp(static_cast<double>(logits[static_cast<size_t>(b) * C + c]) - m);
                }
                const double log_sum_exp = std::log(sum_exp);
                const int t = targets[b];
                const double log_prob = static_cast<double>(logits[static_cast<size_t>(b) * C + t]) - m - log_sum_exp;
                total += -log_prob;
            }
            return reduction_mean ? total / batch : total;
        }

    }  // namespace

    // Batch>1, classes>1; mean reduction. The primary contract for the training
    // drivers (MicroTrainer/TransformerTrainer evaluate mean CE every step).
    TEST_F(LossFunctionsTest, DeviceCrossEntropyLossMatchesFp64Reference) {
        CrossEntropyConfig config;
        config.num_classes = 10;
        config.reduction_mean = true;

        const int batch = 6;
        const int C = 10;
        std::vector<float> logits(static_cast<size_t>(batch) * C);
        std::vector<int> targets(batch);
        std::mt19937 rng(20260823);
        std::normal_distribution<float> dist(0.0f, 2.0f);
        for (size_t i = 0; i < logits.size(); ++i) {
            logits[i] = dist(rng);
        }
        for (int b = 0; b < batch; ++b) {
            targets[b] = static_cast<int>(rng() % C);
        }
        const double expected = host_ce_loss_reference(logits, targets, batch, C, true);

        cuda::memory::Buffer<float> d_logits(logits.size());
        cuda::memory::Buffer<int> d_targets(targets.size());
        cuda::memory::Buffer<float> d_loss(1);
        d_logits.copy_from(logits.data(), logits.size());
        d_targets.copy_from(targets.data(), targets.size());

        EXPECT_NO_THROW(cross_entropy_loss_device(d_logits.data(), d_targets.data(), d_loss.data(), batch, C, config, stream_));
        float dev = 0.0f;
        d_loss.copy_to(&dev, 1);

        // fp32 reduction-order drift vs the fp64 reference (same convention as
        // the existing kernel-parity tests: 1e-4 absorbs block-sum reordering).
        EXPECT_NEAR(static_cast<float>(expected), dev, 1e-4f);
    }

    // Single-row edge (batch == 1) — the degenerate case the trainers can hit at
    // batch*seq == 1; must still produce the exact single-row loss.
    TEST_F(LossFunctionsTest, DeviceCrossEntropyLossSingleRow) {
        CrossEntropyConfig config;
        config.num_classes = 8;
        config.reduction_mean = true;

        const int batch = 1;
        const int C = 8;
        std::vector<float> logits {3.0f, -1.0f, 0.5f, 2.2f, -4.0f, 1.1f, 0.0f, 0.7f};
        std::vector<int> targets {2};
        const double expected = host_ce_loss_reference(logits, targets, batch, C, true);

        cuda::memory::Buffer<float> d_logits(logits.size());
        cuda::memory::Buffer<int> d_targets(targets.size());
        cuda::memory::Buffer<float> d_loss(1);
        d_logits.copy_from(logits.data(), logits.size());
        d_targets.copy_from(targets.data(), targets.size());

        EXPECT_NO_THROW(cross_entropy_loss_device(d_logits.data(), d_targets.data(), d_loss.data(), batch, C, config, stream_));
        float dev = 0.0f;
        d_loss.copy_to(&dev, 1);

        EXPECT_NEAR(static_cast<float>(expected), dev, 1e-4f);
    }

    // Sum reduction (reduction_mean == false) must return the raw summed loss,
    // matching the host cross_entropy_loss convention.
    TEST_F(LossFunctionsTest, DeviceCrossEntropyLossSumReduction) {
        CrossEntropyConfig config;
        config.num_classes = 6;
        config.reduction_mean = false;

        const int batch = 3;
        const int C = 6;
        std::vector<float> logits(static_cast<size_t>(batch) * C);
        std::vector<int> targets(batch);
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.5f);
        for (size_t i = 0; i < logits.size(); ++i) {
            logits[i] = dist(rng);
        }
        for (int b = 0; b < batch; ++b) {
            targets[b] = static_cast<int>(rng() % C);
        }
        const double expected = host_ce_loss_reference(logits, targets, batch, C, false);

        cuda::memory::Buffer<float> d_logits(logits.size());
        cuda::memory::Buffer<int> d_targets(targets.size());
        cuda::memory::Buffer<float> d_loss(1);
        d_logits.copy_from(logits.data(), logits.size());
        d_targets.copy_from(targets.data(), targets.size());

        EXPECT_NO_THROW(cross_entropy_loss_device(d_logits.data(), d_targets.data(), d_loss.data(), batch, C, config, stream_));
        float dev = 0.0f;
        d_loss.copy_to(&dev, 1);

        EXPECT_NEAR(static_cast<float>(expected), dev, 1e-4f);
    }

    // accuracy_device must count rows whose argmax class == target, matching the
    // host top-1 loop the training drivers previously ran on copied-back logits.
    // Includes a tie (row with two equal max logits): host and device argmax both
    // pick the earliest max, so the count must agree exactly.
    TEST_F(LossFunctionsTest, DeviceAccuracyMatchesHostArgmax) {
        const int batch = 5;
        const int C = 4;
        // Row 0: target 1 == argmax(3.0 at class 1)  -> correct
        // Row 1: target 2, argmax class 3            -> wrong
        // Row 2: tie between class 0 and class 1 (both 2.0); earliest is class 0;
        //        target 0 -> correct
        // Row 3: all equal; earliest max is class 0; target 0 -> correct
        // Row 4: target 3 == argmax(9.0 at class 3)  -> correct
        std::vector<float> logits {0.0f, 3.0f, 0.0f, 0.5f, 1.0f, 0.0f, 0.0f, 4.0f, 2.0f, 2.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 0.0f, 0.5f, 9.0f};
        std::vector<int> targets {1, 2, 0, 0, 3};

        int host_correct = 0;
        for (int b = 0; b < batch; ++b) {
            const float* row = logits.data() + static_cast<size_t>(b) * C;
            int argmax = 0;
            for (int c = 1; c < C; ++c) {
                if (row[c] > row[argmax]) {
                    argmax = c;
                }
            }
            if (argmax == targets[b]) {
                ++host_correct;
            }
        }

        cuda::memory::Buffer<float> d_logits(logits.size());
        cuda::memory::Buffer<int> d_targets(targets.size());
        cuda::memory::Buffer<float> d_correct(1);
        d_logits.copy_from(logits.data(), logits.size());
        d_targets.copy_from(targets.data(), targets.size());

        EXPECT_NO_THROW(accuracy_device(d_logits.data(), d_targets.data(), d_correct.data(), batch, C, stream_));
        float dev_correct = -1.0f;
        d_correct.copy_to(&dev_correct, 1);

        EXPECT_EQ(static_cast<float>(host_correct), dev_correct);
    }

    // C > 256: with more classes than the block size, every thread holds a
    // nonzero partial in the grid-stride max/exp-sum loops, so the block
    // reductions actually combine multiple partials (the C<=10 tests above are
    // all single-thread-row partials). Locks the multi-iteration path.
    TEST_F(LossFunctionsTest, DeviceCrossEntropyLossLargeClassCount) {
        CrossEntropyConfig config;
        config.num_classes = 1024;
        config.reduction_mean = true;

        const int batch = 3;
        const int C = 1024;
        std::vector<float> logits(static_cast<size_t>(batch) * C);
        std::vector<int> targets(batch);
        std::mt19937 rng(777);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < logits.size(); ++i) {
            logits[i] = dist(rng);
        }
        for (int b = 0; b < batch; ++b) {
            targets[b] = static_cast<int>(rng() % C);
        }
        const double expected = host_ce_loss_reference(logits, targets, batch, C, true);

        cuda::memory::Buffer<float> d_logits(logits.size());
        cuda::memory::Buffer<int> d_targets(targets.size());
        cuda::memory::Buffer<float> d_loss(1);
        d_logits.copy_from(logits.data(), logits.size());
        d_targets.copy_from(targets.data(), targets.size());

        EXPECT_NO_THROW(cross_entropy_loss_device(d_logits.data(), d_targets.data(), d_loss.data(), batch, C, config, stream_));
        float dev = 0.0f;
        d_loss.copy_to(&dev, 1);

        EXPECT_NEAR(static_cast<float>(expected), dev, 1e-4f);
    }

    // Non-uniform logits: a target pinned to the max-logit class must give a
    // small loss (< 1), a target on the min-logit class a large loss (> 2) —
    // sanity that the device path computes log-probabilities, not something
    // else, before the strict fp64 parity above.
    TEST_F(LossFunctionsTest, DeviceCrossEntropyLossDistinguishesTargets) {
        CrossEntropyConfig config;
        config.num_classes = 3;
        config.reduction_mean = true;

        const int batch = 2;
        const int C = 3;
        // Row 0: target 0 is the max class. Row 1: target 1 is the min class.
        std::vector<float> logits {10.0f, 0.0f, 0.0f, 0.0f, -10.0f, 0.0f};
        std::vector<int> targets {0, 1};

        cuda::memory::Buffer<float> d_logits(logits.size());
        cuda::memory::Buffer<int> d_targets(targets.size());
        cuda::memory::Buffer<float> d_loss(1);
        d_logits.copy_from(logits.data(), logits.size());
        d_targets.copy_from(targets.data(), targets.size());

        EXPECT_NO_THROW(cross_entropy_loss_device(d_logits.data(), d_targets.data(), d_loss.data(), batch, C, config, stream_));
        float dev = 0.0f;
        d_loss.copy_to(&dev, 1);

        // mean of (loss~1e-10 for row0, loss~10 for row1) ≈ 5.
        EXPECT_GT(dev, 1.0f);
        EXPECT_LT(dev, 9.0f);
    }

    // Regression: cross_entropy_loss_device() checked batch_size and num_classes but never the
    // targets, which it uses as an index into the row (row[t]). The host cross_entropy_loss()
    // validates exactly this ("target 99 at index 2 is outside [0, 3)"), so the same inputs
    // that threw on the host returned a plausible-looking loss from the device entry point.
    // Measured before the fix: batch 4, C 3, targets {0, 1, 99, 0} gave 1.223612 with
    // cudaDeviceSynchronize returning cudaSuccess, where an in-range batch gives 1.098612.
    //
    // The read is out of bounds: row[99] on a 3-class row. It is not reliably a crash, which is
    // what makes it worth pinning -- the loss looks like a measurement.
    TEST_F(LossFunctionsTest, DeviceCrossEntropyRejectsOutOfRangeTargets) {
        const int batch = 4;
        const int C = 3;
        CrossEntropyConfig config;
        config.num_classes = C;

        std::vector<float> logits(static_cast<size_t>(batch) * C, 0.5f);

        // Each case: one target outside [0, C) at a different position, plus the two ways it can
        // be outside (too large, negative).
        const std::vector<std::vector<int>> bad_targets = {
            {0, 1, 99, 0},  // too large, in the middle
            {0, 1, C, 0},   // exactly one past the last valid class
            {-1, 1, 2, 0},  // negative, first row
            {0, 1, 2, -7},  // negative, last row
        };

        for (const auto& targets : bad_targets) {
            cuda::memory::Buffer<float> d_logits(logits.size());
            cuda::memory::Buffer<int> d_targets(targets.size());
            cuda::memory::Buffer<float> d_loss(1);
            d_logits.copy_from(logits.data(), logits.size());
            d_targets.copy_from(targets.data(), targets.size());

            EXPECT_THROW(cross_entropy_loss_device(d_logits.data(), d_targets.data(), d_loss.data(), batch, C, config, stream_), std::invalid_argument)
                << "targets with an out-of-range class must be rejected, not used as an index";
            // Leave the device clean for the next case regardless of the outcome.
            (void)cudaStreamSynchronize(stream_);
            (void)cudaGetLastError();
        }

        // The same batch with in-range targets still works and still gives the exact value.
        std::vector<int> good {0, 1, 2, 0};
        cuda::memory::Buffer<float> d_logits(logits.size());
        cuda::memory::Buffer<int> d_targets(good.size());
        cuda::memory::Buffer<float> d_loss(1);
        d_logits.copy_from(logits.data(), logits.size());
        d_targets.copy_from(good.data(), good.size());
        EXPECT_NO_THROW(cross_entropy_loss_device(d_logits.data(), d_targets.data(), d_loss.data(), batch, C, config, stream_));
        float dev = 0.0f;
        d_loss.copy_to(&dev, 1);
        // Uniform 0.5 logits over 3 classes: every target has loss ln(3).
        EXPECT_NEAR(dev, 1.0986123f, 1e-4f) << "the validation must not disturb the valid path";
    }

    // accuracy_device() shares the unchecked index: it compares targets[b] against an argmax,
    // so an out-of-range class silently counted the row as wrong instead of being reported.
    TEST_F(LossFunctionsTest, DeviceAccuracyRejectsOutOfRangeTargets) {
        const int batch = 4;
        const int C = 3;

        std::vector<float> logits(static_cast<size_t>(batch) * C, 0.5f);
        std::vector<int> targets {0, 1, 2, 99};

        cuda::memory::Buffer<float> d_logits(logits.size());
        cuda::memory::Buffer<int> d_targets(targets.size());
        cuda::memory::Buffer<float> d_correct(1);
        d_logits.copy_from(logits.data(), logits.size());
        d_targets.copy_from(targets.data(), targets.size());

        EXPECT_THROW(accuracy_device(d_logits.data(), d_targets.data(), d_correct.data(), batch, C, stream_), std::invalid_argument);
        (void)cudaStreamSynchronize(stream_);
        (void)cudaGetLastError();
    }

}  // namespace cuda::neural::loss::test
