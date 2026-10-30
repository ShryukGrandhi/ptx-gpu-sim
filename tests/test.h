// Minimal self-registering test framework (no external dependencies).
#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace testing {

struct Failure {
    std::string msg;
};
struct Skip {
    std::string reason;
};

struct Case {
    std::string name;
    std::function<void()> fn;
};

inline std::vector<Case>& registry() {
    static std::vector<Case> r;
    return r;
}

struct Registrar {
    Registrar(const char* name, std::function<void()> fn) { registry().push_back({name, std::move(fn)}); }
};

// Paths passed on the command line.
inline std::string& kernel_dir() { static std::string s = "kernels"; return s; }
inline std::string& nvcc_ptx() { static std::string s; return s; }

}  // namespace testing

#define TEST_CAT2(a, b) a##b
#define TEST_CAT(a, b) TEST_CAT2(a, b)
#define TEST(name)                                                                  \
    static void name();                                                             \
    static testing::Registrar TEST_CAT(reg_, name)(#name, name);                    \
    static void name()

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::ostringstream os_;                                                                \
            os_ << __FILE__ << ":" << __LINE__ << ": CHECK(" #cond ") failed";                     \
            throw testing::Failure{os_.str()};                                                     \
        }                                                                                          \
    } while (0)

#define CHECK_EQ(a, b)                                                                             \
    do {                                                                                           \
        auto va_ = (a);                                                                            \
        auto vb_ = (b);                                                                            \
        if (!(va_ == vb_)) {                                                                       \
            std::ostringstream os_;                                                                \
            os_ << __FILE__ << ":" << __LINE__ << ": CHECK_EQ(" #a ", " #b ") failed: " << va_     \
                << " vs " << vb_;                                                                  \
            throw testing::Failure{os_.str()};                                                     \
        }                                                                                          \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                                      \
    do {                                                                                           \
        double va_ = (a), vb_ = (b);                                                               \
        if (std::fabs(va_ - vb_) > (tol)) {                                                        \
            std::ostringstream os_;                                                                \
            os_ << __FILE__ << ":" << __LINE__ << ": CHECK_NEAR(" #a ", " #b ") failed: " << va_   \
                << " vs " << vb_;                                                                  \
            throw testing::Failure{os_.str()};                                                     \
        }                                                                                          \
    } while (0)

#define CHECK_THROWS(expr, Type, substr)                                                           \
    do {                                                                                           \
        bool thrown_ = false;                                                                      \
        try {                                                                                      \
            expr;                                                                                  \
        } catch (const Type& e_) {                                                                 \
            thrown_ = true;                                                                        \
            if (std::string(e_.what()).find(substr) == std::string::npos) {                        \
                std::ostringstream os_;                                                            \
                os_ << __FILE__ << ":" << __LINE__ << ": exception message '" << e_.what()         \
                    << "' does not contain '" << substr << "'";                                    \
                throw testing::Failure{os_.str()};                                                 \
            }                                                                                      \
        }                                                                                          \
        if (!thrown_) {                                                                            \
            std::ostringstream os_;                                                                \
            os_ << __FILE__ << ":" << __LINE__ << ": expected " #Type " from " #expr;              \
            throw testing::Failure{os_.str()};                                                     \
        }                                                                                          \
    } while (0)

#define SKIP(reason) throw testing::Skip{reason}
