#include "test.h"

#include <exception>

// Usage: ptxsim_tests [kernel_dir] [--nvcc-ptx file.ptx] [--filter substring]
int main(int argc, char** argv) {
    std::string filter;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--nvcc-ptx" && i + 1 < argc) testing::nvcc_ptx() = argv[++i];
        else if (a == "--filter" && i + 1 < argc) filter = argv[++i];
        else testing::kernel_dir() = a;
    }
    int passed = 0, failed = 0, skipped = 0;
    for (const auto& c : testing::registry()) {
        if (!filter.empty() && c.name.find(filter) == std::string::npos) continue;
        try {
            c.fn();
            ++passed;
            std::cout << "[ PASS ] " << c.name << "\n";
        } catch (const testing::Skip& s) {
            ++skipped;
            std::cout << "[ SKIP ] " << c.name << ": " << s.reason << "\n";
        } catch (const testing::Failure& f) {
            ++failed;
            std::cout << "[ FAIL ] " << c.name << "\n         " << f.msg << "\n";
        } catch (const std::exception& e) {
            ++failed;
            std::cout << "[ FAIL ] " << c.name << "\n         unexpected exception: " << e.what() << "\n";
        }
    }
    std::cout << "\n" << passed << " passed, " << failed << " failed, " << skipped << " skipped\n";
    return failed == 0 ? 0 : 1;
}
