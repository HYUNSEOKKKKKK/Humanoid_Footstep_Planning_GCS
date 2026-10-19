#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "mosek.h"

namespace py = pybind11;

struct IndexRegistry {
  // 예: 글로벌 인덱스 테이블 (필요한 만큼 확장)
  int n_edge = 0;
  std::vector<int> idx_zL_flat; // E*5
  std::vector<int> idx_zR_flat; // E*5
  // 필요한 제약/슬롯 인덱스도 저장
};

class GcsTask {
public:
  // MSK objects
  MSKenv_t  env  = nullptr;
  MSKtask_t task = nullptr;

  IndexRegistry IDX;
  int nvar = 0, ncon = 0;

  // 내부 캐시
  int E = 0;            // edge 수
  double step_span = 0; // 파라미터 예시

  GcsTask() {
    MSK_makeenv(&env, nullptr);
    MSK_maketask(env, 0, 0, &task);
    // 기본 옵션
    MSK_putintparam(task, MSK_IPAR_LOG, 0);
  }

  ~GcsTask() {
    if (task) MSK_deletetask(&task);
    if (env)  MSK_deleteenv(&env);
  }

  // -------- build: 스파스 패턴 고정 --------
  void build(py::array_t<double> v,                // (N,4,2) vertices (이번 스텁에선 미사용)
             py::array_t<int>    edge,             // (E,2) 0-based
             double               step_span_,
             double               alpha,           // 미니모델에선 미사용
             py::array_t<int>    mode_list,        // (E,) or (something,) 미사용
             py::array_t<double> x_ref,            // (2,) 미사용
             py::array_t<double> y_ref)            // (2,) 미사용
  {
    // E 계산
    auto edge_r = edge.unchecked<2>();
    E = edge_r.shape(0);
    step_span = step_span_;

    // 변수: zL[e,0:5], zR[e,0:5] => 총 10*E (미니 모델에선 2*E만 써도 됨)
    // 여기서는 계약을 맞추기 위해 5개씩 할당
    nvar = 10 * E;
    MSK_appendvars(task, nvar);

    // 인덱스 테이블 구축
    IDX.n_edge = E;
    IDX.idx_zL_flat.resize(E*5);
    IDX.idx_zR_flat.resize(E*5);
    for (int e=0; e<E; ++e) {
      for (int k=0; k<5; ++k) {
        int idxL = e*10 + k;
        int idxR = e*10 + 5 + k;
        IDX.idx_zL_flat[e*5 + k] = idxL;
        IDX.idx_zR_flat[e*5 + k] = idxR;
        // 기본 바운드: free
        MSK_putvarbound(task, idxL, MSK_BK_FR, -MSK_INFINITY, MSK_INFINITY);
        MSK_putvarbound(task, idxR, MSK_BK_FR, -MSK_INFINITY, MSK_INFINITY);
      }
    }

    // === 미니 “배선 테스트” 제약 만들기 ===
    // (1) 선형 등식: zL[e,0] - zR[e,0] = 0
    MSK_appendcons(task, E);
    for (int e=0; e<E; ++e) {
      int row = e;
      int jL0 = IDX.idx_zL_flat[e*5 + 0];
      int jR0 = IDX.idx_zR_flat[e*5 + 0];
      // a[row, jL0] = 1, a[row, jR0] = -1
      MSK_putaij(task, row, jL0,  1.0);
      MSK_putaij(task, row, jR0, -1.0);
      MSK_putconbound(task, row, MSK_BK_FX, 0.0, 0.0);
    }
    ncon = E;

    // (2) SOC: || [ zL[e,0]-zR[e,0], zL[e,1]-zR[e,1] ] ||_2 <= step_span
    // 표준 2차원 원뿔: (t, x, y) with t >= ||(x,y)||
    // 여기서 t = step_span - (zL0 - zR0) 를 쓰기보단, 헷지 없이 auxiliary var t_e 추가 권장
    // 간단화를 위해 여기서는 cone에 직접 (t_e, x, y)을 사용
    // 변수 t_e 추가:
    int t_base = nvar;          // 보조변수 시작
    MSK_appendvars(task, E);    // t_e 하나씩
    for (int e=0; e<E; ++e) {
      int t_idx = t_base + e;
      MSK_putvarbound(task, t_idx, MSK_BK_LO, 0.0, MSK_INFINITY); // t >= 0
    }
    nvar += E;

    // cone: [ t_e, x, y ] in QUAD
    // x = zL0 - zR0, y = zL1 - zR1, 그리고 추가로 t_e <= step_span
    // t_e <= step_span 는 별도 선형 제약으로 처리
    for (int e=0; e<E; ++e) {
      int idx_t = t_base + e;
      int idx_x = IDX.idx_zL_flat[e*5 + 0];
      int idx_y = IDX.idx_zL_flat[e*5 + 1];
      int idx_xr= IDX.idx_zR_flat[e*5 + 0];
      int idx_yr= IDX.idx_zR_flat[e*5 + 1];

      // quadaxis: (t, x, y) -> x,y는 선형 조합이라 MSK_appendcone만으로는 직접 차분 표현이 안 됨
      // => 어피니티로 표현하려면 AF(affine conic) 필요. 간단히 보이기 위해 “변수 치환”용 보조변수 x_e,y_e 도 추가
      // x_e = zL0 - zR0, y_e = zL1 - zR1
    }
    // 간단/명료를 위해 x_e,y_e도 추가
    int xe_base = nvar;
    int ye_base = nvar + E;
    MSK_appendvars(task, 2*E);
    for (int e=0; e<E; ++e) {
      int xe = xe_base + e;
      int ye = ye_base + e;
      MSK_putvarbound(task, xe, MSK_BK_FR, -MSK_INFINITY, MSK_INFINITY);
      MSK_putvarbound(task, ye, MSK_BK_FR, -MSK_INFINITY, MSK_INFINITY);
    }
    nvar += 2*E;

    // x_e = zL0 - zR0, y_e = zL1 - zR1  (E개 등식 추가)
    MSK_appendcons(task, 2*E);
    for (int e=0; e<E; ++e) {
      int rowx = ncon + e;
      int rowy = ncon + E + e;

      int xe = xe_base + e;
      int ye = ye_base + e;

      int jL0 = IDX.idx_zL_flat[e*5 + 0];
      int jL1 = IDX.idx_zL_flat[e*5 + 1];
      int jR0 = IDX.idx_zR_flat[e*5 + 0];
      int jR1 = IDX.idx_zR_flat[e*5 + 1];

      // rowx: xe - zL0 + zR0 = 0
      MSK_putaij(task, rowx, xe, 1.0);
      MSK_putaij(task, rowx, jL0, -1.0);
      MSK_putaij(task, rowx, jR0,  1.0);
      MSK_putconbound(task, rowx, MSK_BK_FX, 0.0, 0.0);

      // rowy: ye - zL1 + zR1 = 0
      MSK_putaij(task, rowy, ye, 1.0);
      MSK_putaij(task, rowy, jL1, -1.0);
      MSK_putaij(task, rowy, jR1,  1.0);
      MSK_putconbound(task, rowy, MSK_BK_FX, 0.0, 0.0);
    }
    ncon += 2*E;

    // cone QUAD: (t_e, x_e, y_e)
    for (int e=0; e<E; ++e) {
      MSKint32t sub[3] = { (MSKint32t)(t_base + e),
                           (MSKint32t)(xe_base + e),
                           (MSKint32t)(ye_base + e) };
      MSK_appendcone(task, MSK_CT_QUAD, 0.0, 3, sub);
    }

    // t_e <= step_span
    MSK_appendcons(task, E);
    for (int e=0; e<E; ++e) {
      int row = ncon + e;
      int t_idx = t_base + e;
      MSK_putaij(task, row, t_idx, 1.0);
      MSK_putconbound(task, row, MSK_BK_UP, -MSK_INFINITY, step_span);
    }
    ncon += E;

//    // 목적: sum(zL[e,0] + zR[e,0]) 최소 (단순 선형)
//    for (int e=0; e<E; ++e) {
//      MSK_putcj(task, IDX.idx_zL_flat[e*5 + 0], 1.0);
//      MSK_putcj(task, IDX.idx_zR_flat[e*5 + 0], 1.0);
//    }
//    MSK_putobjsense(task, MSK_OBJECTIVE_SENSE_MINIMIZE);

    // 교체: t_e 합 최소화
    for (int e=0; e<E; ++e) {
      int t_idx = t_base + e;
      MSK_putcj(task, t_idx, 1.0);
    }
    MSK_putobjsense(task, MSK_OBJECTIVE_SENSE_MINIMIZE);
  }

  // -------- set_params: 수치 업데이트만 --------
  void set_params(py::dict params) {
    // 예시: M_large, start/target을 이용해 바운드 업데이트, 목적 가중치 업데이트 등
    // params["M_large"], ["start_L"], ... 키 접근
    // 필요 시 MSK_putaij / MSK_putconbound / MSK_putvarbound / MSK_putcj 호출
    // 본 미니모델에선 특별히 갱신할 게 없으니 no-op 가능
    if (params.contains("M_large")) {
      // placeholder
    }
  }

  // -------- solve --------
  py::dict solve(int log_level) {
    MSK_putintparam(task, MSK_IPAR_LOG, log_level ? 1 : 0);

    double t0 = now_ms();
    MSKrescodee trm;
    MSK_optimizetrm(task, &trm);
    double t1 = now_ms();

    // 상태 체크
    MSKsolstae solsta;
    MSKprostae prosta;
    MSK_getsolutionstatus(task, MSK_SOL_ITR, &prosta, &solsta);

    // 목적값
    double obj = 0.0;
    MSK_getprimalobj(task, MSK_SOL_ITR, &obj);

    // 변수값
    std::vector<double> xx(nvar, 0.0);
    MSK_getxx(task, MSK_SOL_ITR, xx.data());

    // zL/zR 재구성: (E,5)
    py::array_t<double> zL({E,5});
    py::array_t<double> zR({E,5});
    auto zLw = zL.mutable_unchecked<2>();
    auto zRw = zR.mutable_unchecked<2>();
    for (int e=0; e<E; ++e) {
      for (int k=0; k<5; ++k) {
        zLw(e,k) = xx[IDX.idx_zL_flat[e*5 + k]];
        zRw(e,k) = xx[IDX.idx_zR_flat[e*5 + k]];
      }
    }

    py::dict out;
    out["obj"] = obj;
    out["zL"]  = zL;
    out["zR"]  = zR;
    out["time_ms"] = (t1 - t0);
    return out;
  }

private:
  static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
  }
};

PYBIND11_MODULE(gcs_task, m) {
  py::class_<GcsTask>(m, "GcsTask")
    .def(py::init<>())
    .def("build", &GcsTask::build,
         py::arg("v"), py::arg("edge"), py::arg("step_span"),
         py::arg("alpha"), py::arg("mode_list"),
         py::arg("x_ref"), py::arg("y_ref"))
    .def("set_params", &GcsTask::set_params)
    .def("solve", &GcsTask::solve, py::arg("log_level")=0);
}
