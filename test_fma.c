#include <math.h>
#include <stdbool.h>
#include <stdio.h>

struct fma_test_case_d
{
    double a, b, c, expected;
} static const cases_d[] = {
        {0x1p1000, 0x1p1000, -INFINITY, -INFINITY},
        {-0x1.4f8ac19291ffap1023, 0x1.39c33c8d39b7p-1025, 0x1.ee11f685e2e12p-1, 0x1.2071b0283f156p-1},
        {0x1.0000000000008p500, 0x1.1p500, 0x1p-1074, 0x1.1000000000009p1000},
        {0x1.0000000000001p500, 0x1.8p500, -0x1p-1074, 0x1.8000000000001p1000},
        {0x0.ffffffep513, 0x1.0000002p511, -0x1p-1074, 0x1.fffffffffffffp1023},
};

struct fma_test_case_f
{
    float a, b, c, expected;
} static const cases_f[] = {
        {0x1p100f, 0x1p100f, -INFINITY, -INFINITY},
        {0x1.fffffep23f, 0x1.000004p28f, 0x1.fcp5f, 0x1.000002p52f},
        {0x1.84ae3p125f, 0x1.6p-141f, 0x1p-149f, 0x1.0b37c2p-15f},
        {0x1.00001p50f, 0x1.1p50f, 0x1p-149f, 0x1.100012p100f},
        {0x1.000002p50f, 0x1.8p50f, -0x1p-149f, 0x1.800002p100f},
        {0x1.83bd78p4f, -0x1.cp118f, -0x1.344108p-2f, -0x1.5345cap123f},
        {0x1.060000p7f, 0x1.340488p7f, 0x1.310e42p4f, 0x1.3b88e6p14f},
};

int main(void)
{
#if defined(FP_FAST_FMA)
    puts("FP_FAST_FMA is defined.");
#endif
#if defined(FP_FAST_FMAF)
    puts("FP_FAST_FMAF is defined.");
#endif
#if defined(__FMA__)
    puts("__FMA__ is defined.");
#endif
#if defined(__FMA4__)
    puts("__FMA4__ is defined.");
#endif
    {
        bool all_tests_passed = true;
        for (size_t i = 0; i < sizeof(cases_d) / sizeof(cases_d[0]); ++i) {
            double a = cases_d[i].a;
            double b = cases_d[i].b;
            double c = cases_d[i].c;
            double expected = cases_d[i].expected;
            double result = fma(a, b, c);
            bool same = isnan(expected) ? isnan(result) : expected == result && signbit(expected) == signbit(result);
            if (!same) {
                all_tests_passed = false;
                printf("Testcase[%zd]: fma(%a, %a, %a): Expected %a, but got %a.\n", i, a, b, c, expected, result);
            }
        }
        if (all_tests_passed) {
            puts("All tests passed for fma(double, double, double).");
        }
    }
    {
        bool all_tests_passed = true;
        for (size_t i = 0; i < sizeof(cases_f) / sizeof(cases_f[0]); ++i) {
            float a = cases_f[i].a;
            float b = cases_f[i].b;
            float c = cases_f[i].c;
            float expected = cases_f[i].expected;
            float result = fmaf(a, b, c);
            bool same = isnan(expected) ? isnan(result) : expected == result && signbit(expected) == signbit(result);
            if (!same) {
                all_tests_passed = false;
                printf("Testcase[%zd]: fmaf(%a, %a, %a): Expected %a, but got %a.\n", i, a, b, c, expected, result);
            }
        }
        if (all_tests_passed) {
            puts("All tests passed for fmaf(float, float, float).");
        }
    }
}
