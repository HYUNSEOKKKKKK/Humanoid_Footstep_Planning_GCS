import sys, os
# 빌드 산출물(.so) 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cpp', 'build'))

import msk_capi
print("MSK version:", msk_capi.mosek_version())
