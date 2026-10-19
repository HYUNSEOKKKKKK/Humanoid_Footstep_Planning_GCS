#include <pybind11/pybind11.h>
#include "mosek.h"
namespace py = pybind11;

std::string mosek_version() {
  MSKenv_t env = nullptr;
  MSK_makeenv(&env, nullptr);   // 환경은 만들어두어도 OK (다른 API에 필요할 수 있음)

  MSKint32t a = 0, b = 0, c = 0;
  // ✅ env 인자 없이 호출해야 합니다.
  MSK_getversion(&a, &b, &c);

  MSK_deleteenv(&env);

  char buf[64];
  snprintf(buf, sizeof(buf), "%d.%d.%d", (int)a, (int)b, (int)c);
  return std::string(buf);
}

PYBIND11_MODULE(msk_capi, m) {
  m.doc() = "MSK C Task API pybind11 bridge (minimal)";
  m.def("mosek_version", &mosek_version);
}
