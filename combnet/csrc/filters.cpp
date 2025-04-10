#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <vector>

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

namespace combnet {

void single_comb_iir_cpu(
    const double f0,
    const double a,
    const int64_t sr,
    at::Tensor& y
) {
    // TODO: add checks
    at::Tensor y_contig = y.contiguous();
    float* y_ptr = y.data_ptr<float>();
    int delay = sr/f0;
    for (int64_t i=delay; i<y.numel(); i++) {
        y_ptr[i] += a * y_ptr[i-delay];
    }
}

// void single_fractional_comb_iir_cpu(
//     const double f0,
//     const double a,
//     const int64_t sr,
//     at::Tensor& y
// ) {
//     // TODO: add checks
//     at::Tensor y_contig = y.contiguous();
//     float* y_ptr = y.data_ptr<float>();
//     float delay = sr/f0;
//     for (int64_t i=delay; i<y.numel(); i++) {
//         y_ptr[i] += a * y_ptr[i-delay];
//     }
// }

// Defines the operators
TORCH_LIBRARY(combnet, m) {
    m.def("single_comb_iir(float f0, float a, int sr, Tensor(a!) y) -> ()");
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(combnet, CPU, m) {
  m.impl("single_comb_iir", &single_comb_iir_cpu);
}

}