#include <cuda/quantize/fp8_gemm.hpp>
#include <cuda/quantize/fp8_types.hpp>

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

namespace nova {
    namespace quantize {
        namespace test {

            class FP8GEMMTest : public ::testing::Test {
            protected:
                void SetUp() override {
                    cudaError_t err = cudaSetDevice(0);
                    if (err != cudaSuccess) {
                        GTEST_SKIP() << "CUDA not available";
                    }
                }

                static constexpr float kTol = 0.05f;

                float relative_error(float original, float recovered) {
                    if (std::abs(original) < 1e-6f) {
                        return 0.0f;
                    }
                    return std::abs(original - recovered) / std::abs(original);
                }

                void fill_random_fp32(std::vector<float>& vec, float scale = 1.0f) {
                    for (auto& v : vec) {
                        v = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
                    }
                }

                std::vector<FP8E4M3> convert_to_fp8e4m3(const std::vector<float>& vec) {
                    std::vector<FP8E4M3> result(vec.size());
                    for (size_t i = 0; i < vec.size(); ++i) {
                        result[i] = FP8E4M3(vec[i]);
                    }
                    return result;
                }

                void reference_gemm(const float* a, const float* b, float* c, int m, int k, int n) {
                    for (int i = 0; i < m; ++i) {
                        for (int j = 0; j < n; ++j) {
                            float sum = 0.0f;
                            for (int p = 0; p < k; ++p) {
                                sum += a[i * k + p] * b[p * n + j];
                            }
                            c[i * n + j] = sum;
                        }
                    }
                }
            };

            TEST_F(FP8GEMMTest, SmallMatrixMultiplication) {
                int m = 4, k = 4, n = 4;

                std::vector<float> a(m * k), b(k * n), c_ref(m * n);

                a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};

                b = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};

                reference_gemm(a.data(), b.data(), c_ref.data(), m, k, n);

                auto a_fp8 = convert_to_fp8e4m3(a);
                auto b_fp8 = convert_to_fp8e4m3(b);

                std::vector<float> c_gpu(m * n, 0.0f);

                float *d_a, *d_b, *d_c;
                cudaMalloc(&d_a, m * k * sizeof(FP8E4M3));
                cudaMalloc(&d_b, k * n * sizeof(FP8E4M3));
                cudaMalloc(&d_c, m * n * sizeof(float));

                cudaMemcpy(d_a, a_fp8.data(), m * k * sizeof(FP8E4M3), cudaMemcpyHostToDevice);
                cudaMemcpy(d_b, b_fp8.data(), k * n * sizeof(FP8E4M3), cudaMemcpyHostToDevice);

                FP8GEMM::Config config;
                config.scale_a = 1.0f;
                config.scale_b = 1.0f;
                config.scale_out = 1.0f;

                FP8GEMM::forward(reinterpret_cast<const FP8E4M3*>(d_a), reinterpret_cast<const FP8E4M3*>(d_b), d_c, m, k, n, config, 0);

                cudaMemcpy(c_gpu.data(), d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost);

                for (int i = 0; i < m * n; ++i) {
                    float rel_err = relative_error(c_ref[i], c_gpu[i]);
                    EXPECT_LE(rel_err, kTol) << "Mismatch at index " << i;
                }

                cudaFree(d_a);
                cudaFree(d_b);
                cudaFree(d_c);
            }

            TEST_F(FP8GEMMTest, IdentityMatrix) {
                int m = 8, k = 8, n = 8;

                std::vector<float> a(m * k), b(k * n), c_ref(m * n);

                for (int i = 0; i < m; ++i) {
                    for (int j = 0; j < k; ++j) {
                        a[i * k + j] = (i == j) ? 1.0f : 0.0f;
                    }
                }

                for (int i = 0; i < k; ++i) {
                    for (int j = 0; j < n; ++j) {
                        b[i * n + j] = (i == j) ? 1.0f : 0.0f;
                    }
                }

                reference_gemm(a.data(), b.data(), c_ref.data(), m, k, n);

                auto a_fp8 = convert_to_fp8e4m3(a);
                auto b_fp8 = convert_to_fp8e4m3(b);

                std::vector<float> c_gpu(m * n, 0.0f);

                float *d_a, *d_b, *d_c;
                cudaMalloc(&d_a, m * k * sizeof(FP8E4M3));
                cudaMalloc(&d_b, k * n * sizeof(FP8E4M3));
                cudaMalloc(&d_c, m * n * sizeof(float));

                cudaMemcpy(d_a, a_fp8.data(), m * k * sizeof(FP8E4M3), cudaMemcpyHostToDevice);
                cudaMemcpy(d_b, b_fp8.data(), k * n * sizeof(FP8E4M3), cudaMemcpyHostToDevice);

                FP8GEMM::Config config;

                FP8GEMM::forward(reinterpret_cast<const FP8E4M3*>(d_a), reinterpret_cast<const FP8E4M3*>(d_b), d_c, m, k, n, config, 0);

                cudaMemcpy(c_gpu.data(), d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost);

                for (int i = 0; i < m; ++i) {
                    for (int j = 0; j < n; ++j) {
                        float expected = (i == j) ? 1.0f : 0.0f;
                        float rel_err = relative_error(expected, c_gpu[i * n + j]);
                        EXPECT_LE(rel_err, kTol) << "Mismatch at [" << i << "," << j << "]";
                    }
                }

                cudaFree(d_a);
                cudaFree(d_b);
                cudaFree(d_c);
            }

            // Random inputs exercise the whole quantize-then-multiply path. The
            // reference is built from the QUANTIZED inputs, not the originals.
            //
            // The two differ, and by a lot: E4M3 keeps 3 mantissa bits, so each
            // operand is rounded by up to ~3% and the relative error of a
            // k=16 dot product is heavy-tailed near cancellation -- measured
            // worst case over 20000 random pairs, 553. It is bounded only by
            // |sum| >= 1e-4, the guard below. Comparing against the unquantized
            // product therefore cannot be made to pass with any tolerance that
            // still means something, which is why this case sat DISABLED_ and
            // reported a "2.12 > 2" failure when run.
            //
            // Against the quantized reference the kernel is exact: measured max
            // relative error 0.000000, i.e. the GEMM computes sum(q(a)*q(b)) in
            // fp32 with no drift of its own.
            TEST_F(FP8GEMMTest, RandomMatrices) {
                int m = 16, k = 16, n = 16;

                std::vector<float> a(m * k), b(k * n), c_ref(m * n);

                fill_random_fp32(a, 1.0f);
                fill_random_fp32(b, 1.0f);

                auto a_fp8 = convert_to_fp8e4m3(a);
                auto b_fp8 = convert_to_fp8e4m3(b);

                // The GEMM never sees a[] and b[]; it sees their FP8 encodings. The
                // reference has to start from the same values or it is comparing the
                // kernel against a different problem.
                std::vector<float> a_q(a.size()), b_q(b.size());
                for (size_t i = 0; i < a.size(); ++i) {
                    a_q[i] = static_cast<float>(a_fp8[i]);
                }
                for (size_t i = 0; i < b.size(); ++i) {
                    b_q[i] = static_cast<float>(b_fp8[i]);
                }
                reference_gemm(a_q.data(), b_q.data(), c_ref.data(), m, k, n);

                std::vector<float> c_gpu(m * n, 0.0f);

                float *d_a, *d_b, *d_c;
                cudaMalloc(&d_a, m * k * sizeof(FP8E4M3));
                cudaMalloc(&d_b, k * n * sizeof(FP8E4M3));
                cudaMalloc(&d_c, m * n * sizeof(float));

                cudaMemcpy(d_a, a_fp8.data(), m * k * sizeof(FP8E4M3), cudaMemcpyHostToDevice);
                cudaMemcpy(d_b, b_fp8.data(), k * n * sizeof(FP8E4M3), cudaMemcpyHostToDevice);

                FP8GEMM::Config config;

                FP8GEMM::forward(reinterpret_cast<const FP8E4M3*>(d_a), reinterpret_cast<const FP8E4M3*>(d_b), d_c, m, k, n, config, 0);

                cudaMemcpy(c_gpu.data(), d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost);

                float max_rel_err = 0.0f;
                for (int i = 0; i < m * n; ++i) {
                    if (std::abs(c_ref[i]) < 1e-4f) {
                        continue;
                    }
                    float rel_err = relative_error(c_ref[i], c_gpu[i]);
                    max_rel_err = std::max(max_rel_err, rel_err);
                }

                // The measured value is 0 against this reference; 1e-5 leaves room for
                // fp32 accumulation order without accepting a real deviation.
                EXPECT_LE(max_rel_err, 1e-5f) << "Max relative error: " << max_rel_err;

                cudaFree(d_a);
                cudaFree(d_b);
                cudaFree(d_c);
            }

            TEST_F(FP8GEMMTest, ConfigScaling) {
                int m = 4, k = 4, n = 4;

                std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f};

                std::vector<float> b = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};

                auto a_fp8 = convert_to_fp8e4m3(a);
                auto b_fp8 = convert_to_fp8e4m3(b);

                float *d_a, *d_b, *d_c;
                cudaMalloc(&d_a, m * k * sizeof(FP8E4M3));
                cudaMalloc(&d_b, k * n * sizeof(FP8E4M3));
                cudaMalloc(&d_c, m * n * sizeof(float));

                cudaMemcpy(d_a, a_fp8.data(), m * k * sizeof(FP8E4M3), cudaMemcpyHostToDevice);
                cudaMemcpy(d_b, b_fp8.data(), k * n * sizeof(FP8E4M3), cudaMemcpyHostToDevice);

                {
                    FP8GEMM::Config config;
                    config.scale_a = 1.0f;
                    config.scale_b = 1.0f;
                    config.scale_out = 1.0f;

                    FP8GEMM::forward(reinterpret_cast<const FP8E4M3*>(d_a), reinterpret_cast<const FP8E4M3*>(d_b), d_c, m, k, n, config, 0);

                    std::vector<float> c1(m * n);
                    cudaMemcpy(c1.data(), d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost);

                    for (int i = 0; i < m; ++i) {
                        for (int j = 0; j < n; ++j) {
                            if (i == j) {
                                EXPECT_NEAR(c1[i * n + j], static_cast<float>(i + 1), 1.0f);
                            }
                        }
                    }
                }

                {
                    FP8GEMM::Config config;
                    config.scale_a = 2.0f;
                    config.scale_b = 0.5f;
                    config.scale_out = 1.0f;

                    FP8GEMM::forward(reinterpret_cast<const FP8E4M3*>(d_a), reinterpret_cast<const FP8E4M3*>(d_b), d_c, m, k, n, config, 0);

                    std::vector<float> c2(m * n);
                    cudaMemcpy(c2.data(), d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost);
                }

                cudaFree(d_a);
                cudaFree(d_b);
                cudaFree(d_c);
            }

            TEST_F(FP8GEMMTest, WorkspaceSizeQuery) {
                int m = 128, k = 256, n = 128;

                FP8GEMM::Config config;

                size_t workspace = FP8GEMM::get_workspace_size(m, k, n, config);

                EXPECT_GE(workspace, static_cast<size_t>(m * k * sizeof(FP8E4M3)));
                EXPECT_GE(workspace, static_cast<size_t>(k * n * sizeof(FP8E4M3)));
                EXPECT_GE(workspace, static_cast<size_t>(m * n * sizeof(float)));
            }

            // Regression: backward() had no test, and both gradients were wrong. For C = A(m x k) * B(k x n)
            // with upstream gradient G(m x n):
            //   grad_a = G * B^T   (m x k)
            //   grad_b = A^T * G   (k x n)
            // The implementation computed grad_a as G * B (correct only when B happens to be symmetric)
            // and grad_b as G^T * A (transposed). Measured with A = B = [[1,2],[3,4]], G = ones:
            // grad_b came back [[4,6],[4,6]] where [[4,4],[6,6]] is correct.
            //
            // The shapes here are deliberately non-square and B is deliberately asymmetric, because a
            // square symmetric B makes every one of those mistakes look correct.
            TEST_F(FP8GEMMTest, BackwardProducesBothGradientsForAnAsymmetricB) {
                constexpr int m = 2, k = 3, n = 2;
                const float a_values[m * k] = {1, 2, 3, 4, 5, 6};
                const float b_values[k * n] = {1, 2, 3, 4, 5, 6};
                const float g_values[m * n] = {1, 1, 1, 1};

                std::vector<FP8E4M3> a(a_values, a_values + m * k);
                std::vector<FP8E4M3> b(b_values, b_values + k * n);
                std::vector<FP8E4M3> g(g_values, g_values + m * n);

                FP8E4M3* d_a = nullptr;
                FP8E4M3* d_b = nullptr;
                FP8E4M3* d_g = nullptr;
                float* d_grad_a = nullptr;
                float* d_grad_b = nullptr;
                ASSERT_EQ(cudaMalloc(&d_a, m * k), cudaSuccess);
                ASSERT_EQ(cudaMalloc(&d_b, k * n), cudaSuccess);
                ASSERT_EQ(cudaMalloc(&d_g, m * n), cudaSuccess);
                ASSERT_EQ(cudaMalloc(&d_grad_a, m * k * sizeof(float)), cudaSuccess);
                ASSERT_EQ(cudaMalloc(&d_grad_b, k * n * sizeof(float)), cudaSuccess);
                ASSERT_EQ(cudaMemcpy(d_a, a.data(), m * k, cudaMemcpyHostToDevice), cudaSuccess);
                ASSERT_EQ(cudaMemcpy(d_b, b.data(), k * n, cudaMemcpyHostToDevice), cudaSuccess);
                ASSERT_EQ(cudaMemcpy(d_g, g.data(), m * n, cudaMemcpyHostToDevice), cudaSuccess);

                nova::quantize::FP8GEMM::backward(d_g, d_a, d_b, d_grad_a, d_grad_b, m, k, n, nova::quantize::FP8GEMM::Config {});
                ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

                std::vector<float> grad_a(m * k, 0.0f);
                std::vector<float> grad_b(k * n, 0.0f);
                ASSERT_EQ(cudaMemcpy(grad_a.data(), d_grad_a, m * k * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
                ASSERT_EQ(cudaMemcpy(grad_b.data(), d_grad_b, k * n * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
                cudaFree(d_a);
                cudaFree(d_b);
                cudaFree(d_g);
                cudaFree(d_grad_a);
                cudaFree(d_grad_b);

                // Reference computed from the decoded values, so this checks the GEMM and not the format.
                auto decode = [](const FP8E4M3& v) { return static_cast<float>(v); };
                for (int i = 0; i < m; ++i) {
                    for (int j = 0; j < k; ++j) {
                        double expected = 0.0;
                        for (int p = 0; p < n; ++p) {
                            expected += static_cast<double>(decode(g[static_cast<size_t>(i) * n + p])) * decode(b[static_cast<size_t>(j) * n + p]);
                        }
                        EXPECT_NEAR(grad_a[static_cast<size_t>(i) * k + j], expected, 1e-3) << "grad_a at [" << i << "," << j << "]";
                    }
                }
                for (int i = 0; i < k; ++i) {
                    for (int j = 0; j < n; ++j) {
                        double expected = 0.0;
                        for (int p = 0; p < m; ++p) {
                            expected += static_cast<double>(decode(a[static_cast<size_t>(p) * k + i])) * decode(g[static_cast<size_t>(p) * n + j]);
                        }
                        EXPECT_NEAR(grad_b[static_cast<size_t>(i) * n + j], expected, 1e-3) << "grad_b at [" << i << "," << j << "]";
                    }
                }
            }
        }  // namespace test
    }  // namespace quantize
}  // namespace nova
