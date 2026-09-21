// BAD candidate: missing semicolon and undefined function
extern "C" void launch_kernel(
    float* x, float* weight, float* y,
    int batch, int hidden, float eps
) {
    this_will_fail_to_compile(x, weight)
}
