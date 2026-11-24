# hrc_ws 재현 가이드 (초안)

새 Ubuntu 22.04 PC에서 **hrc_ws + realtime_smpl**(sensing_start, human_dynamics)을 이 PC와 같은 상태로 만드는 절차.
위에서 아래로 순서대로 따라 하면 된다. 값은 기준 PC(2026-11-18)에서 읽어 온 것이고, 확인하지 못한 항목은 **[확인 필요]**로 표시했다.

**필요 없는 것:** 로봇 제어 `indy_ws`, `~/python/physpt`, gRPC/Protobuf, Pinocchio.
ISO/TS 15066 표(`server/iso_15066.json`), 관절 가동 범위(`src/body_models/joint_rom/joint_ROM.json`),
`server/camera_pose.yaml`은 이 저장소에 들어 있다. `server/physics.pkl`은 SMPL 모델에서 만든 파일이라 넣지 않았다(7장).

기준 PC 요약

| 항목 | 값 |
|---|---|
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0-138-generic |
| GPU / 드라이버 | NVIDIA GeForce RTX 2080 Ti / 580.178.04 |
| CUDA toolkit | 12.4 (`/usr/local/cuda-12.4`, nvcc V12.4.99) |
| PyTorch | 2.4.0+cu124, torchvision 0.19.0+cu124, cuDNN 9.1 |
| ROS 2 | Humble (`ros-humble-ros-base` 0.10.0, `ros-humble-cv-bridge` 3.2.1) |
| Python | 시스템 python3 3.10, 패키지는 `pip install --user` (`~/.local`) |
| Azure Kinect | SDK 1.3.0 (`libk4a1.3`), Body Tracking 1.0.0 (`libk4abt1.0`) |
| OpenCV | 4.2.0 + contrib, 소스 빌드 → `/usr/local` |

목차
0. 시작 전 확인
1. 전제
2. Azure Kinect SDK 1.3 / Body Tracking 1.0
3. apt 패키지
4. 소스 받기
5. OpenCV 4.2.0 + contrib 소스 빌드
6. Python 패키지
7. 모델·가중치
8. 빌드
9. 환경 변수
10. 실행과 카메라 자세(roll·pitch·yaw·t)
11. 동작 확인 체크리스트
12. 문제 해결
13. 확인 필요 항목

---

## 0. 시작 전 확인

설치 명령을 실행하기 전에 아래를 모두 확인한다. 하나라도 기준 PC와 다르면 진행 방법을 먼저 정한다.

### 0.1 GPU 세대

```bash
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
```

| compute_cap | 세대 (예) | 판단 |
|---|---|---|
| 7.5 이하 | Turing(RTX 20, GTX 16) 이하 | 이 문서 그대로 진행 |
| 8.0 이상 | Ampere(RTX 30, A6000), Ada(RTX 40) 이후 | **Body Tracking 1.0이 동작하지 않을 수 있다**(1장 경고). 2.5의 BT 동작 확인을 통과하기 전에는 오래 걸리는 단계(5장 이후)를 시작하지 않는다 |

### 0.2 conda 끄고 시스템 Python 확인

ROS 2 Humble과 이 문서의 모든 `pip install --user`는 **시스템 python3 3.10**(`/usr/bin/python3`)을 기준으로 한다.
conda(base 포함)가 켜져 있으면 `python3`가 conda의 Python을 가리켜 다른 곳에 설치된다.

```bash
conda deactivate                       # base까지 모두 끈다 (여러 번 필요할 수 있음)
# 새 터미널에서도 자동으로 켜지지 않게 하려면: conda config --set auto_activate_base false
which python3                          # /usr/bin/python3 이어야 한다
python3 --version                      # Python 3.10.x
```

이후 모든 단계는 이 상태의 터미널에서 실행한다.

### 0.3 `~/.local`의 다른 프로젝트 영향

6장은 `pip install --user`로 `~/.local`에 **고정 버전**(torch 2.4.0+cu124, numpy 1.26.4, matplotlib 3.5.3 등)을 설치한다.
`~/.local`은 같은 사용자의 시스템 Python 3.10을 쓰는 모든 프로젝트가 함께 쓰므로, 이미 설치된 판이 바뀌면 다른 프로젝트가 깨질 수 있다.
또 pip로 설치한 `opencv-python`은 `/usr/local`의 cv2 4.2(5장)와 충돌한다(6.1).

```bash
python3 -m pip list --user 2>/dev/null | grep -iE "^(torch|torchvision|numpy|scipy|matplotlib|opencv|pillow|smplx|chumpy) "
python3 -m pip freeze --user > ~/pip_user_backup_$(date +%F).txt   # 되돌릴 때 쓸 기록
```

위 목록에 requirements.txt와 다른 판이 있으면, 다른 프로젝트에서 그 판을 쓰는지 확인하고 진행 여부를 정한다.

### 0.4 bionic(18.04) Microsoft 저장소 금지

`packages.microsoft.com/ubuntu/18.04` 저장소를 apt 소스에 **등록하지 않는다**(2장 경고, 12장). 이미 있으면 지우고 `sudo apt update`.

```bash
grep -rh "packages.microsoft.com/ubuntu/18.04" /etc/apt/sources.list /etc/apt/sources.list.d/ || echo "OK: bionic repo 없음"
```

### 0.5 이미 설치된 다른 버전의 Kinect SDK

```bash
dpkg -l | grep -iE "k4a|k4abt"                  # libk4a1.4, libk4abt1.1, k4a-tools 1.4 등이 있는지
```

**다른 버전의 k4a(1.4 등)가 이미 설치되어 있으면, 지우기 전에 멈추고 판단한다. 1.3과 1.4는 함께 설치할 수 없다.**

- 패키지에 `Conflicts`가 선언되어 있지는 않지만 같은 경로의 파일을 갖고 있어서, 함께 설치하면 dpkg가 파일 덮어쓰기 오류로 멈춘다
  (2026-11-18, `.deb` 파일 목록 비교):
  - `libk4a1.3` ↔ `libk4a1.4`: `/usr/lib/x86_64-linux-gnu/libk4a.so`, `libk4arecord.so`
  - `libk4a1.3-dev` ↔ `libk4a1.4-dev`: `/usr/include/k4a/*`, `/usr/include/k4arecord/*` 등 18개
  - `libk4abt1.0-dev` ↔ `libk4abt1.1-dev`: `/usr/include/k4abt*.h`, `/usr/lib/cmake/k4abt/*` 등 9개
  - `libk4abt1.1`은 `libk4a1.4 (= 1.4.1)`에, `libk4abt1.0`은 `libk4a1.3`에 의존한다. 즉 BT 판이 SDK 판을 정한다.
- 판단할 것:
  - 이 PC에서 1.4/1.1에 링크된 다른 프로그램(다른 워크스페이스의 빌드 결과, `k4aviewer` 등)을 찾는다.
    1.4를 지우면 그 프로그램은 다시 빌드할 때까지 실행되지 않는다(`ldd <파일> | grep k4a`로 확인).
  - 나중에 되돌릴 수 있게 지금 설치된 `.deb`가 남아 있는지 확인한다.
  - Ampere 이후 GPU(0.1)라면, BT 1.0으로 바꾸기 전에 이미 설치된 1.1로 2.5의 동작 확인을 먼저 해 본다.

#### 0.5.1 설치된 libk4a가 Microsoft 원본인지 확인 (Orbbec 래퍼 덮어쓰기)

Orbbec 카메라(Femto 등)용 **OrbbecSDK K4A Wrapper**는 Microsoft와 같은 이름의 `libk4a.so.1.4.1`, `libk4arecord.so.1.4.1`을 만든다.
이 파일을 `/usr/lib/x86_64-linux-gnu`에 복사하면 dpkg에는 `libk4a1.4`가 그대로 설치된 것으로 보이지만, 실제로는 Orbbec 래퍼가 불려
**Azure Kinect를 찾지 못한다**(장치 0개). `dpkg -l`만으로는 구분되지 않는다.

판별:

```bash
dpkg --verify libk4a1.4 libk4a1.4-dev libk4abt1.1 k4a-tools   # 출력이 없어야 원본. "??5??????  …/libk4a.so.1.4.1"이면 덮어쓰기
ls /usr/lib/x86_64-linux-gnu | grep -iE "OrbbecSDK|^libdepthengine"   # 출력이 없어야 한다(Microsoft depth engine은 libk4a1.4/ 아래에 있다)
```

실행 로그로도 알 수 있다. `k4abt_check`(2.5)나 k4a를 쓰는 프로그램을 실행했을 때 아래 같은 줄이 나오면 Orbbec 래퍼다(Microsoft libk4a는 이런 로그를 내지 않는다).

```
[..][info][..][Context.cpp:73] Work directory=..., SDK version=v1.10.5-...
[..][info][..][DeviceManager.cpp:15] Current found device(s): (0)
```

Orbbec 래퍼가 덮어쓴 경우 되돌리기. **삭제하지 말고 백업 폴더로 옮긴다.**

1. 백업: `/usr/lib/x86_64-linux-gnu`의 `libk4a*`, `libk4arecord*`, `libdepthengine*`, `libOrbbecSDK*`를 `cp -a`로 복사하고 sha256을 남긴다.
2. Microsoft 원본 재설치: `sudo apt install --reinstall ./libk4a1.4_1.4.1_amd64.deb`(2.1과 같은 pool의 `.deb`. `dpkg --verify`가 깨끗한 `-dev`, `k4a-tools`는 재설치할 필요 없음).
3. 패키지 소유가 아닌 Orbbec 파일을 백업 폴더로 옮긴다. 옮기기 전에 각각 `dpkg -S <파일>`이 "no path found"인지 확인한다.
   `libOrbbecSDK.so*`, `libk4arecord.so.1.4.0`, 그리고 Microsoft depth engine(`libk4a1.4/libdepthengine.so.2.0`)을 가릴 수 있는 `libdepthengine.so`, `libdepthengine.so.2.0`.
4. `sudo ldconfig` 후 위 판별 명령을 다시 실행하고 2.5를 실행한다.

- Orbbec udev 규칙(`/etc/udev/rules.d/99-obsensor-libusb.rules`, vendor `2bc5`만 해당)은 Azure Kinect에 영향이 없어 그대로 둔다.
- 되돌린 뒤에는 Orbbec 카메라를 이 시스템 라이브러리로 쓸 수 없다. 필요하면 홈의 래퍼 폴더를 `LD_LIBRARY_PATH`로 지정해 그 프로그램에서만 쓴다.

### 0.6 진행 순서

1. **Kinect SDK 1.3 / BT 1.0 설치**(2.1~2.4)
2. **BT 동작 확인**(2.5). 통과하지 못하면 여기서 멈추고 원인을 정한다
3. 그다음 오래 걸리는 단계: apt 패키지(3장) → 소스(4장) → OpenCV 빌드(5장) → Python 패키지(6장) → 모델(7장) → 빌드(8장)

BT가 이 PC에서 동작하지 않으면 5~6장의 빌드·설치는 쓸모가 없으므로, 반드시 2.5를 먼저 통과시킨다.

## 1. 전제

1. Ubuntu 22.04 (jammy), x86_64.
2. NVIDIA 드라이버. 기준 PC는 **580.178.04**. `nvidia-smi`가 동작해야 한다(드라이버를 설치하거나 업데이트한 뒤에는 재부팅).
3. CUDA toolkit 12.4 (`/usr/local/cuda-12.4`). PyTorch wheel은 자체 CUDA 런타임(cu124)을 쓰지만,
   `minimal_pytorch_rasterizer` CUDA 확장 빌드(6.2)에 `nvcc` 12.4가 필요하다.
4. ROS 2 Humble: `ros-humble-ros-base`, `ros-humble-cv-bridge`.
5. GPU 세대: Body Tracking 1.0은 CUDA 10.0 / cuDNN 7을 자체 동봉한다(`/usr/lib/libcudart.so.10.0`, `libcudnn.so.7`, `libonnxruntime.so.1.0.0`, 모델 `/usr/bin/dnn_model_2_0.onnx`).

> ⚠️ **BT 1.0은 CUDA 10 / cuDNN 7 내장. Ampere 이후 GPU(RTX 30/40, A6000)에서는 동작하지 않을 수 있음.
> 기준 PC는 2080 Ti(Turing).**

## 2. Azure Kinect SDK 1.3 / Body Tracking 1.0

> ⚠️ **bionic(18.04) Microsoft 저장소를 apt 소스에 등록하지 말 것.** 등록된 채로 두면 jammy의 unixODBC
> 패키지(`libodbc2`, `unixodbc-common` …)와 Microsoft 저장소의 ODBC 패키지가 충돌해 `dpkg`가 멈춘다.
> 아래처럼 `.deb` 4개만 직접 받아 설치한다.

### 2.1 .deb 받기 (packages.microsoft.com pool, 2026-11-18 16:07에 4개 모두 HTTP 200 확인)

```bash
mkdir -p ~/k4a_debs && cd ~/k4a_debs
B=https://packages.microsoft.com/ubuntu/18.04/prod/pool/main
wget $B/libk/libk4a1.3/libk4a1.3_1.3.0_amd64.deb             #   4,859,944 B
wget $B/libk/libk4a1.3-dev/libk4a1.3-dev_1.3.0_amd64.deb     #      42,236 B
wget $B/libk/libk4abt1.0/libk4abt1.0_1.0.0_amd64.deb         # 434,898,864 B
wget $B/libk/libk4abt1.0-dev/libk4abt1.0-dev_1.0.0_amd64.deb #      10,786 B
# (선택) k4aviewer 등 도구: $B/k/k4a-tools/k4a-tools_1.3.0_amd64.deb  (4,335,694 B)
```


### 2.2 설치

```bash
cd ~/k4a_debs
sudo apt install ./libk4a1.3_1.3.0_amd64.deb ./libk4a1.3-dev_1.3.0_amd64.deb \
                 ./libk4abt1.0_1.0.0_amd64.deb ./libk4abt1.0-dev_1.0.0_amd64.deb
```

- 설치 중 EULA 동의 화면이 뜬다(`libk4a1.3`, `libk4abt1.0`).
  realtime_smpl의 원본 Docker(`docker/accept_eula/Dockerfile`)는 `libk4a1.3`의 동의를 debconf로 미리 넣는다.
  ```bash
  echo 'libk4a1.3 libk4a1.3/accepted-eula-hash string 0f5d5c5de396e4fee4c0753a21fee0c1ed726cf0316204edda484f08cb266d76' | sudo debconf-set-selections
  echo 'libk4a1.3 libk4a1.3/accept-eula boolean true' | sudo debconf-set-selections
  ```
  같은 파일의 주석에 따르면 `libk4abt1.0`은 이 방법이 통하지 않아 화면에서 직접 동의해야 한다.
  위 debconf 방법을 jammy에서 `.deb`로 설치할 때 써 보지는 않았다 **[확인 필요]**.
- 의존성은 jammy 기본 패키지로 충족된다(`libgl1`, `libudev1`, `libx11-6`, `libgomp1`, `libxcursor1`, `libxinerama1`, `libxrandr2`, `libxxf86vm1`).
- 기준 PC에는 예전에 설치했던 `libk4a1.4`, `libk4abt1.1`이 `rc`(설정만 남음) 상태로 있다. 새 PC에서는 1.3 / 1.0만 설치한다.

### 2.3 설치 후 확인

```bash
dpkg -l | grep -E "libk4a|libk4abt"          # libk4a1.3 1.3.0, libk4a1.3-dev, libk4abt1.0 1.0.0, libk4abt1.0-dev
apt-cache policy unixodbc                    # Candidate 가 jammy(-updates/-security)를 가리켜야 한다
grep -rh "packages.microsoft.com/ubuntu/18.04" /etc/apt/sources.list /etc/apt/sources.list.d/ || echo "OK: bionic repo 없음"
ldd /usr/lib/libk4abt.so.1.0 | grep "not found" || echo "OK: BT 라이브러리 의존성 충족"
```

기준 PC 값: `unixodbc` Candidate `2.3.9-5ubuntu0.1` (jammy-updates/universe).

### 2.4 장치 권한 (udev, 필요)

`libk4a`/`libk4abt` 패키지는 udev 규칙을 설치하지 않는다(`dpkg -L`로 확인).
기준 PC의 `/etc/udev/rules.d/99-k4a.rules`는 수동으로 복사한 파일이다(root, 2025-12-29). Azure Kinect SDK 저장소의 `scripts/99-k4a.rules`
(`~/realtime_smpl/extern/azure-sdk/scripts/99-k4a.rules`)와 바이트 단위로 같다. 이 규칙이 없으면 일반 사용자로 장치를 열 수 없다.

```bash
sudo tee /etc/udev/rules.d/99-k4a.rules >/dev/null <<'EOF'
BUS!="usb", ACTION!="add", SUBSYSTEM!=="usb_device", GOTO="k4a_logic_rules_end"
ATTRS{idVendor}=="045e", ATTRS{idProduct}=="097a", MODE="0666", GROUP="plugdev"
ATTRS{idVendor}=="045e", ATTRS{idProduct}=="097b", MODE="0666", GROUP="plugdev"
ATTRS{idVendor}=="045e", ATTRS{idProduct}=="097c", MODE="0666", GROUP="plugdev"
ATTRS{idVendor}=="045e", ATTRS{idProduct}=="097d", MODE="0666", GROUP="plugdev"
ATTRS{idVendor}=="045e", ATTRS{idProduct}=="097e", MODE="0666", GROUP="plugdev"
LABEL="k4a_logic_rules_end"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger   # 그 뒤 Kinect USB 재연결
```

### 2.5 BT 동작 확인 (`tools/k4abt_check.c`)

3장 이후로 넘어가기 전에, 이 PC에서 Body Tracking이 실제로 도는지 확인한다(0.6).
`tools/k4abt_check.c`는 장치 열기 → 깊이 카메라 시작 → tracker 생성 → 프레임 N개 처리 순으로 진행하고,
프레임마다 사람 수와 처리 시간(enqueue → pop)을 출력한다. 실패한 단계 이름과 종료 코드로 어디서 막혔는지 알 수 있다.

```bash
# Kinect 연결, conda deactivate 상태, 카메라 앞에 사람 1명
mkdir -p ~/hrc_ws/build
gcc -O2 -Wall -o ~/hrc_ws/build/k4abt_check ~/hrc_ws/tools/k4abt_check.c -lk4a -lk4abt
~/hrc_ws/build/k4abt_check            # 기본 30프레임. 프레임 수 지정: k4abt_check 100
```

출력 예(형식):

```
[k4abt_check] devices: 1
[k4abt_check] device opened, serial 000123456789
[k4abt_check] cameras started (NFOV_UNBINNED, 30 fps, color off)
[k4abt_check] tracker created in 2140.3 ms
[k4abt_check] frame   0: bodies 1, 812.4 ms   (첫 프레임은 모델 준비로 느림)
[k4abt_check] frame   1: bodies 1, 41.2 ms
...
[k4abt_check] OK: 30 frames, frames with body 29, mean 39.8 ms (excluding frame 0)
```

| 종료 코드 | 막힌 단계 | 먼저 볼 것 |
|---|---|---|
| 0 | 통과 | – |
| 2 | 장치 없음 / 열기 실패 | USB 연결, udev 규칙(2.4), 다른 프로세스가 장치를 잡고 있는지(12장) |
| 3 | 캘리브레이션 / 카메라 시작 실패 | 전원·USB 3.0 포트 |
| 4 | tracker 생성 실패 | GPU 세대(0.1, Ampere 이후에서 BT 1.0), `ldd /usr/lib/libk4abt.so.1.0`, 모델 `/usr/bin/dnn_model_2_0.onnx`, 터미널에 찍힌 libk4abt 오류 |
| 5 | 프레임 처리 실패(capture / enqueue / pop) | 터미널에 찍힌 libk4abt 오류 |

- **통과 기준:** 종료 코드 0, 사람이 서 있는 동안 `bodies 1`, 0번 이후 프레임 처리 시간이 수십 ms 수준.
  기준 PC(2080 Ti)의 처리 시간은 아직 이 프로그램으로 재지 않았다 **[확인 필요]**.
- 오류 로그를 더 보려면 `K4ABT_LOG_LEVEL=i K4A_LOG_LEVEL=i ~/hrc_ws/build/k4abt_check`.
- 실행 파일은 `build/`(git 무시)에 만든다.

## 3. apt 패키지

```bash
sudo apt install build-essential cmake git pkg-config unzip wget gh \
  python3-pip python3-yaml python3-colcon-common-extensions python3-vcstool \
  libturbojpeg libturbojpeg0-dev libjpeg-turbo8-dev \
  nlohmann-json3-dev libeigen3-dev libusb-1.0-0-dev libsoundio-dev \
  libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
```

- 기준 PC 버전: `libturbojpeg` 2.1.2, `nlohmann-json3-dev` 3.10.5, `libeigen3-dev` 3.4.0, `libusb-1.0-0-dev` 1.0.25, `libsoundio-dev` 2.0.0.
- `libtbb-dev` 2021.5.0도 설치되어 있지만 OpenCV에서는 쓰지 않는다(5장).
- OpenCV 빌드용 GTK/FFmpeg/V4L 개발 패키지 목록은 기준 PC의 실제 설치 목록과 대조하지 않았다 **[확인 필요]**.
- **nlohmann_json:** pyk4a C 확장(`#include <nlohmann/json.hpp>`)만 쓴다. pyk4a `setup.py`의 include 순서가
  `/usr/include` → `/usr/local/include`라서 apt 3.10.5가 먼저 잡힌다.
  기준 PC의 `/usr/local`에는 3.12.0(`realtime_smpl/extern/nlohmann_json` 커밋 `30b28175e46827153d8a51d75540ef888aa83d8b`)도 설치되어 있다.
  apt 판만으로 pyk4a를 빌드해 보지는 않았다 **[확인 필요]**.

## 4. 소스 받기

이 저장소(hrc_ws-public)를 `~/hrc_ws` 에 받는다. 사람 인식이 import 하는 extern 패키지(`patched_smplx`, `smplx_kinect`, `smplx_optimization`,
`minimal_pytorch_rasterizer`, `pyk4a`)는 [rgbd-kinect-pose](https://github.com/rmbashirov/rgbd-kinect-pose) 의 것을 `~/realtime_smpl` 에 둔다(6.2).

```bash
cd ~
git clone https://github.com/ukhyeon/hrc_ws-public.git hrc_ws
```

경로 기본값은 `~/hrc_ws`, `~/realtime_smpl`이다. 다른 위치에 두면 9장의 `HRC_WS_DIR`, `REALTIME_SMPL_DIR`을 설정한다.

## 5. OpenCV 4.2.0 + contrib 소스 빌드 (→ /usr/local)

pyk4a C 확장이 `/usr/local/include/opencv4`와 `/usr/local/lib/libopencv_{core,calib3d,imgproc}.so.4.2`에 링크하고,
Python `cv2`도 이 빌드(`/usr/local/lib/python3.10/dist-packages/cv2`)를 쓴다.
apt의 `python3-opencv`(4.5.4)가 함께 설치되어 있어도 `/usr/local` 쪽이 먼저 import된다. ROS `cv_bridge`는 apt의 OpenCV 4.5로 링크되어 있어서, 한 프로세스에 두 판이 함께 올라간다(기준 PC에서 정상 동작).

```bash
python3 -m pip install --user numpy==1.26.4   # cv2 를 numpy 1.x 로 빌드하기 위해 먼저 설치 (6.1 참고)
cd ~/realtime_smpl/extern                     # opencv 소스와 zip은 git에 없음(.gitignore)
wget -O opencv.zip         https://github.com/opencv/opencv/archive/4.2.0.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.2.0.zip
unzip -q opencv.zip && unzip -q opencv_contrib.zip
mkdir -p opencv-4.2.0/build && cd opencv-4.2.0/build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=$HOME/realtime_smpl/extern/opencv_contrib-4.2.0/modules \
      -D WITH_TBB=OFF -D BUILD_TBB=OFF \
      -D WITH_CUDA=OFF -D WITH_OPENGL=OFF -D WITH_QT=OFF -D WITH_OPENMP=OFF \
      -D WITH_GTK=ON -D WITH_FFMPEG=ON -D WITH_V4L=ON -D WITH_IPP=ON -D WITH_EIGEN=ON -D WITH_VTK=ON -D WITH_LAPACK=ON \
      -D WITH_JPEG=ON -D BUILD_JPEG=OFF -D WITH_PROTOBUF=ON -D BUILD_PROTOBUF=ON -D BUILD_opencv_dnn=ON \
      -D BUILD_opencv_python3=ON -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
      -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_ENABLE_NONFREE=OFF -D ENABLE_FAST_MATH=OFF \
      -D BUILD_EXAMPLES=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_DOCS=OFF ..
make -j"$(nproc)" && sudo make install && sudo ldconfig
python3 -c "import cv2; print(cv2.__version__, cv2.__file__)"   # 4.2.0 /usr/local/lib/python3.10/dist-packages/...
```

- 옵션은 기준 PC의 `opencv-4.2.0/build/CMakeCache.txt`(2025-12-29)에서 읽은 값이다. 명령줄로 넘긴 것과 기본값을 구분할 수 없어서 핵심 옵션을 모두 적었다.
- **WITH_TBB=OFF / BUILD_TBB=OFF 이유:** Ubuntu 22.04의 oneTBB(2021.5, `libtbb-dev`) 헤더 구조가 OpenCV 4.2.0과 맞지 않는다
  (OpenCV 4.2는 oneTBB에서 제거된 구 TBB 헤더·API를 쓴다). 설치 당시 Notion 기록.
- 기준 PC의 CMakeCache에는 numpy include 경로가 `numpy/_core/include`(numpy 2.x 구조)로 남아 있다. 즉 빌드 당시에는 numpy 2.x였다.
  numpy 2로 빌드한 cv2도 numpy 1.26.4에서 동작한다(기준 PC). 위 순서(1.26.4를 먼저 설치하고 빌드)로 해 보지는 않았다 **[확인 필요]**.
- 기준 PC의 `~/realtime_smpl/extern/opencv.zip`, `opencv_contrib.zip`은 GitHub archive(4.2.0)와 **같은 파일**이다(2026-11-18 sha256 대조):

  | 파일 | 크기 (B) | sha256 |
  |---|---|---|
  | `opencv.zip` (`opencv/opencv/archive/4.2.0.zip`) | 91,666,896 | `55bd939079d141a50fca74bde5b61b339dd0f0ece6320ec76859aaff03c90d9f` |
  | `opencv_contrib.zip` (`opencv/opencv_contrib/archive/4.2.0.zip`) | 62,527,980 | `a4c0b1fb169a68085f7deefdeca8429aafadf3e1798abec0c1a1a7fc83e11a69` |

  받은 뒤 `sha256sum opencv.zip opencv_contrib.zip`으로 확인한다.

## 6. Python 패키지

### 6.1 pip 패키지 (`hrc_ws/requirements.txt` + `requirements-torch-<cu>.txt`)

torch·torchvision은 CUDA toolkit 판에 맞춰 따로 고른다. `minimal_pytorch_rasterizer`(6.2)를 빌드하는 `nvcc`와 torch의 CUDA 판이 같아야 한다.

| CUDA toolkit (`nvcc`) | torch 파일 |
|---|---|
| 12.4 (기준 PC) | `requirements-torch-cu124.txt` |
| 11.8 | `requirements-torch-cu118.txt` |

```bash
cd ~/hrc_ws
CU=cu124                                                   # 11.8이면 cu118
PIP="python3 -m pip install --no-cache-dir --user"         # venv를 쓰면 --user 를 뺀다
# 1) torch + 공통 (kornia, chumpy 제외)
$PIP -r requirements-torch-$CU.txt -r <(grep -vE '^(chumpy|kornia)' requirements.txt)
# 2) kornia 0.3.0: 메타데이터가 torch==1.5.0을 요구해 함께 설치할 수 없다 -> 의존성 없이
$PIP --no-deps kornia==0.3.0
# 3) chumpy 0.70: setup.py가 import pip -> 빌드 격리 없이
$PIP --no-deps --no-build-isolation chumpy==0.70
# 4) chumpy를 numpy 1.26에서 import되게 (아래 참고)
tools/patch_chumpy.sh
```

`requirements.txt`를 pip 한 번에 설치하면 kornia(torch==1.5.0 요구)와 chumpy(빌드 중 `No module named 'pip'`) 때문에 실패한다(2026-11-18 확인).

| 패키지 | 버전 | | 패키지 | 버전 |
|---|---|---|---|---|
| torch | 2.4.0+cu124 (또는 +cu118) | | trimesh | 3.23.5 |
| torchvision | 0.19.0+cu124 (또는 +cu118) | | pyrender | 0.1.39 |
| numpy | 1.26.4 | | PyOpenGL | 3.1.0 |
| scipy | 1.15.3 | | imageio | 2.37.2 |
| matplotlib | 3.5.3 | | freetype-py | 2.5.1 |
| Pillow | 9.5.0 | | nvidia-ml-py3 | 7.352.0 |
| kornia | 0.3.0 | | yacs | 0.1.8 |
| smplx | 0.1.28 | | tqdm | 4.67.1 |
| chumpy | 0.70 | | torchgeometry | 0.1.2 |

- 목록을 만든 방법: 두 노드를 생성하고, 첫 프레임 이후에 만들어지는 모델(Inferer, Filterer, SMPL/SMPL-X 래퍼, 메시 시각화)까지 실제로 생성해 본 뒤, 메모리에 올라온 모듈을 배포 패키지로 대응시켰다. 버전은 모두 기준 PC의 설치 버전이다.
- `cv2`(5장 소스 빌드), PyYAML·psutil·lark·lxml·pydot(ROS가 쓰는 Ubuntu `python3-*`), extern 패키지(6.2)는 목록에 없다.
  **`opencv-python`을 pip로 설치하지 말 것.** `/usr/local`의 cv2 4.2와 충돌한다.
- **numpy 1.26.4(< 2)로 고정하는 이유**
  - matplotlib 3.5.3 등 고정한 일부 wheel이 numpy 1.x ABI로 빌드되어 numpy 2에서 import가 실패한다.
  - extern 코드와 chumpy가 `np.bool` 같은 별칭을 쓰는데, 이 별칭은 numpy 1.24에서 제거됐다(extern 코드는 realtime_smpl 커밋 2942b24에서 이미 고쳤다).
  - numpy 1.x 마지막 판 1.26.4가 torch 2.4.0, OpenCV 4.2와 함께 기준 PC에서 동작한다.
- **chumpy 0.70:** SMPL `.pkl`(`pykinect/body_models/smpl/SMPL_*.pkl`)을 읽을 때 필요하다(`shapedirs`가 chumpy 객체).
  - PyPI 원본은 numpy 1.24 이상에서 **import되지 않는다**: `chumpy/__init__.py` 11행
    `from numpy import bool, int, float, complex, object, unicode, str, nan, inf` → `ImportError: cannot import name 'bool'`.
  - 기준 PC의 `~/.local` 사본은 이 한 줄만 주석 처리되어 있다(2026-11-18 확인, git에 기록되지 않은 수정은 이것뿐).
  - **requirements 설치 후 `tools/patch_chumpy.sh`를 실행한다.** 현재 `python3`(venv를 쓰면 venv를 켠 상태)의 chumpy를 찾아
    0.70인지 확인하고 그 줄을 주석 처리한다. 원본은 `__init__.py.orig`로 남고, 여러 번 실행해도 안전하다. `PYTHON=<python>`으로 대상을 지정할 수 있다.
  - chumpy `setup.py`가 `import pip`를 하므로 새 환경에서는 `--no-build-isolation`이 필요하다(pip 26에서 확인).

### 6.2 realtime_smpl/extern 패키지

hrc_ws가 import하는 것은 `patched_smplx`, `smplx_kinect`, `smplx_optimization`, `minimal_pytorch_rasterizer`, `pyk4a` 5개다.
`face_expression`, `minimal_hand`, `multiprocessing_pipeline`은 hrc_ws 실행에 쓰이지 않는다(realtime_smpl 서버용).

목표는 **editable 설치**다. 소스를 고치면 바로 반영되고, `~/.local`에 수정된 사본이 따로 생기지 않는다.

```bash
cd ~/realtime_smpl/extern
python3 -m pip install --user -e patched_smplx        # install_requires: torchgeometry (6.1에 포함)
python3 -m pip install --user -e smplx_kinect
python3 -m pip install --user -e smplx_optimization
python3 -m pip install --user --no-build-isolation -e minimal_pytorch_rasterizer   # CUDA 확장, 빌드할 때 torch·nvcc 필요
python3 -m pip install --user -e pyk4a                # C 확장 k4a_module: libk4a/libk4abt(2장), turbojpeg·nlohmann(3장),
                                                      # /usr/local OpenCV 4.2(5장) 필요
python3 -B -c "import smplx_kinect, smplx_optimization, patched_smplx, pyk4a, k4a_module, minimal_pytorch_rasterizer as m; print('ok', m.__file__)"
python3 -B -c "from smplx_kinect.common.smplx_vis import VisualizerMeshSMPL; from smplx_kinect.common.body_models import SMPLWrapper; print('vis ok')"
```

- **기준 PC와 다른 점:** 기준 PC는 `minimal_pytorch_rasterizer`만 editable이고, 나머지는 `~/.local`에 사본으로 설치되어 있다.
  pyk4a 는 IMU 읽기를 고친 판([ukhyeon/pyk4a](https://github.com/ukhyeon/pyk4a) 브랜치 `hrc`)을 쓴다.
  editable로 실제 설치해 보지는 않았다 **[확인 필요]**. 특히 `pyk4a`를 editable로 설치했을 때 `k4a_module*.so`가 어디에 생기는지와 그 파일이 git에 잡히지 않는지 **[확인 필요]**.
- 기준 PC의 `/usr/local/lib/python3.10/dist-packages`에는 예전 `face_expression-1.0-py3.10.egg`가 남아 있다. 새 PC에서는 필요 없다.

## 7. 모델·가중치 (원 배포처에서 받기)

모델과 가중치는 이 저장소에 없다. 각 배포처의 라이선스에 동의하고 받아 아래 위치에 둔다.

| 파일 | 놓을 위치 | 쓰는 노드 | 받는 곳 · 라이선스 |
|---|---|---|---|
| CLIFF HR-W48 체크포인트 `hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt` | `~/hrc_ws/src/hrc_body_tracking/hrc_body_tracking/models/cliff_hr48/` | sensing | [CLIFF](https://github.com/huawei-noah/noah-research/tree/master/CLIFF) 의 사전학습 가중치 |
| `smpl_mean_params.npz` | 같은 폴더 | sensing | CLIFF / [SPIN](https://github.com/nkolot/SPIN) 데이터 |
| `neutral.pkl` (PhysPT 가 가공한 SMPL neutral: 37관절 `J_regressor`, `V_regressor`, `kintree_table_J`) | `~/hrc_ws/src/body_models/smpl/neutral.pkl` | sensing, human_dynamics | [PhysPT](https://github.com/zhangy76/PhysPT) README "Model and Data Download" 의 `assets/data/smpl/neutral.pkl`. SMPL 라이선스. 크기 40,900,293 B, SHA-256 `f66b2b95e01fdcb316cba5ff992e46eacfc606b04e2e8b306f923379d4d584a2` |
| SMPL+H `SMPLH_MALE.npz` | `~/hrc_ws/src/body_models/smplh/male.npz` (이름을 바꾼다) | human_dynamics | [MANO / SMPL+H](https://mano.is.tue.mpg.de) (extended SMPL+H). MANO 라이선스 |
| SMPL `SMPL_MALE.pkl`, `SMPL_FEMALE.pkl` | `~/realtime_smpl/pykinect/body_models/smpl/` | sensing | [SMPL](https://smpl.is.tue.mpg.de). SMPL 라이선스 |
| SMPL-X `SMPLX_MALE.npz`, `SMPLX_FEMALE.npz` | `~/realtime_smpl/pykinect/body_models/smplx/` | sensing | [SMPL-X](https://smpl-x.is.tue.mpg.de). SMPL-X 라이선스 |
| `smplx_kinect/04/` (`config.yaml`, `models/net_110000`, `train_sample_azure.pickle`), `pykinect/rob75_val/` (`J.npy`, `s2k_f.npy`, `s2k_m.npy`, `weights.npy`), `pykinect/kintree_kinect`, `shapefit/renat/betas.npy` | `~/realtime_smpl/` 같은 경로 | sensing | [rgbd-kinect-pose](https://github.com/rmbashirov/rgbd-kinect-pose) 데모 데이터 |
| Azure Kinect SDK 1.3 / Body Tracking 1.0 `.deb` | 복사하지 않고 apt 로 설치 (2장) | – | Microsoft (packages.microsoft.com), Microsoft EULA |

`src/body_models/joint_rom/joint_ROM.json`(관절 가동 범위)과 `src/hrc_body_tracking/server/iso_15066.json`(ISO/TS 15066 부위 표)은 이 저장소에 있다.
`src/hrc_body_tracking/server/physics.pkl`(SMPL 위상의 꼭짓점·면 번호를 담은 작은 파일)은 SMPL 모델에서 만든 것이라 넣지 않았다.

## 8. 빌드

```bash
cd ~/hrc_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install           # hrc_interfaces + hrc_body_tracking
source install/setup.bash
```

`--symlink-install`이므로 `src/`의 Python 수정은 다시 빌드하지 않아도 반영된다.
다음 경우에는 다시 빌드한 뒤 **열려 있는 모든 터미널에서** `source install/setup.bash`를 다시 실행한다(또는 새 터미널을 연다).
- `setup.py`(entry point, package data)를 바꿨을 때
- 메시지 정의(`hrc_interfaces`)를 바꿨을 때
- 복사 설치(`--symlink-install` 없이)에서 전환했을 때. 이전 환경으로 source한 터미널은 옛 경로를 계속 쓴다.

## 9. 환경 변수 (`~/.bashrc`, 기준 PC와 같게)

8장의 빌드를 마친 뒤 추가한다. `install/setup.bash`는 빌드 전에는 존재하지 않는다.

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib          # OpenCV 4.2 (cv2, k4a_module)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
export OMP_WAIT_POLICY=Passive     # Kinect Body Tracking 사용 시 CPU 100% 점유 버그 방지 (아래 참고)
export ROS_DOMAIN_ID=30            # 함께 쓰는 모든 노드(로봇 쪽 포함)가 같은 값이어야 함
export ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.bash
source ~/hrc_ws/install/setup.bash
# 선택:
# export K4A_LOG_LEVEL=w           # libk4a 경고를 터미널에 출력 (IMU 큐 경고 확인용)
# export REALTIME_SMPL_DIR=/path   # 기본 ~/realtime_smpl 이면 불필요
# export HRC_WS_DIR=/path          # 기본 ~/hrc_ws 이면 불필요
# export HRC_CAMERA_POSE_FILE=/path  # 기본 src/hrc_body_tracking/server/camera_pose.yaml 이면 불필요
```

`OMP_WAIT_POLICY=Passive`의 근거는 원본 Docker다. realtime_smpl `docker/Dockerfile` 185행(rgbd-kinect-pose도 같음)에
`# prevent 100% cpu utilization bug when using kinect body tracking` 주석과 함께 설정되어 있다.
pyk4a README도 이 값을 rc 파일에 넣으라고 안내한다.

## 10. 실행과 카메라 자세(roll·pitch·yaw·t)

```bash
ros2 run hrc_body_tracking sensing_start     # 터미널 1
ros2 run hrc_body_tracking human_dynamics    # 터미널 2
```

카메라→world 변환은 `cam_R = Rx(roll)·Ry(pitch)·Rz(yaw)`, `p_world = cam_Rᵀ·p_color + t`(world z 위쪽, 바닥 z = 0)이다.

| 값 | 어디서 | 상태 |
|---|---|---|
| roll·pitch | **sensing 시작 때 Kinect IMU로 자동 측정**(0.3초 버린 뒤 2초 평균) | 측정이 검사(\|g\| 9.81±0.3, 정지 상태)를 통과하지 못하거나 장치 핸들이 없으면 **노드가 종료된다**(코드 1). 이유는 FATAL 로그와 `debug_log/camera_pose_<시각>.yaml`에 남는다. 다른 값으로 대신 쓰는 경로는 없다 |
| yaw·t | `src/hrc_body_tracking/server/camera_pose.yaml` | t = 줄자로 잰 카메라 위치 `[-0.83, -2.10, 1.66]`(원점 = 로봇 베이스 중심 아래 바닥). yaw = 0.0으로 **아직 검증하지 않음** |

- sensing은 측정한 roll·pitch와 yaml의 yaw·t를 `/hrc/camera_pose`(transient_local)로 한 번 발행한다.
- human_dynamics는 이 토픽을 받기 전에는 유효질량을 계산하지 않고 5초마다 대기 로그를 남긴다.
- **카메라를 건드리지 않은 상태에서** sensing을 시작한다.
- SMPL 골반 기준점(J0) 보정은 human_dynamics의 계산식 안에서 처리하므로, t에는 실제 카메라 위치를 그대로 쓴다(hrc_ws 커밋 a1a57ef).

### 10.1 yaw·t 검증 (`tools/camera_pose_fit.py`)

카메라를 새로 설치했거나 옮겼으면 바닥 표시 위에 서서 yaw·t를 맞춘다.
두 노드가 실행 중이어야 하고(`/hrc/human_dynamics_state`, `/hrc/human_pose` 사용), `install/setup.bash`를 source한 터미널에서 실행한다.

```bash
cd ~/hrc_ws
python3 tools/camera_pose_fit.py selftest                       # 계산식 자체 점검 (장치 불필요)
python3 tools/camera_pose_fit.py record --mark M1 -1.40 -0.30 --mark M2 -0.80 0.50 \
        --mark M3 -1.80 0.50 --mark M4 -1.40 1.20 --laps 2      # 바닥 표시(world x, y)마다 서서 기록
python3 tools/camera_pose_fit.py fit debug_log/camera_pose_fit/<session>     # yaw·t 추정
python3 tools/camera_pose_fit.py touch --part right_hand --xyz X Y Z --label base_top   # 알려진 점으로 확인
```

- 측정점은 왼발과 오른발 분절 중심의 중점이고, 그 점의 발바닥 위 높이는 체형(betas)으로 계산한다.
- 기록은 `debug_log/camera_pose_fit/<session>/`에 남는다.
- `fit` 결과로 `camera_pose.yaml`의 `yaw_deg`와 `t`를 고친 뒤 두 노드를 다시 시작한다.
- 바닥 표시 좌표는 위 예시 값이다. 실제 실험 공간의 표시 위치에 맞춰 바꾼다 **[확인 필요: 기준 PC에서 아직 검증 측정을 하지 않음]**.

## 11. 동작 확인 체크리스트

- [ ] sensing: `[camera pose] libk4a mapped in this process: 1 [...]`. **1개**여야 한다.
- [ ] sensing: `[camera pose] roll … pitch … (IMU, |g| …, n …); yaw … t … (camera_pose.yaml)`.
      기준 PC 값은 roll 101.02°, pitch 0.59°, |g| 9.83 m/s²(설치 위치가 같을 때). 측정 기록은 `~/hrc_ws/debug_log/camera_pose_<시각>.yaml`에 남는다.
- [ ] human_dynamics: `[camera pose] received /hrc/camera_pose: roll … -- source: imu`. 값이 sensing과 같아야 한다.
      그 전까지는 `still waiting for /hrc/camera_pose …`(5초마다)가 나오는 것이 정상이다.
- [ ] sensing 로그에 `ISO fixed mass not found` 경고가 **없어야** 한다. `Failed to load ISO fixed-mass table`도 없어야 한다.
- [ ] 처리율(기준 PC, RTX 2080 Ti): sensing 약 **20 Hz**(사람 추론 중), human_dynamics 약 **30 Hz**.
      `analysis/realtime_perf/paper_human_{sensing,dynamics}_perf.csv`의 `t_ros_sec` 간격으로 확인한다.
- [ ] `K4A_LOG_LEVEL=w`일 때 IMU 큐 관련 경고(queue, dropped)가 없어야 한다.

## 12. 문제 해결

| 증상 | 원인 / 조치 |
|---|---|
| `apt`/`dpkg`가 ODBC 패키지(`unixodbc`, `libodbc2`, `msodbcsql` 등) 충돌로 멈춤 | Microsoft **bionic(18.04)** 저장소가 apt 소스에 등록되어 있음. 해당 `.list`를 지우고 `sudo apt update`, `apt-cache policy unixodbc`가 jammy를 가리키는지 확인. Kinect SDK는 2.1처럼 `.deb`로만 설치 |
| `Failed to initialize NVML: Driver/library version mismatch`, torch가 CUDA를 못 찾음 | 드라이버를 업데이트한 뒤 재부팅하지 않음 → **재부팅** |
| k4aviewer / Body Tracking이 OpenGL 4.4 관련 오류로 시작하지 않음, 또는 오류 코드 207 | NVIDIA 드라이버의 OpenGL(4.4 이상)로 렌더링하는지 확인(`sudo apt install mesa-utils; glxinfo -B`에서 renderer가 NVIDIA). 원격(SSH/X 포워딩)이 아닌 로컬 디스플레이에서 실행. **[확인 필요: 기준 PC에서 재현·기록 안 됨, error 207의 정확한 출처]** |
| `importlib.metadata.PackageNotFoundError: hrc-body-tracking` 또는 옛 코드가 실행됨 | 빌드 방식이 바뀐 뒤 이전 환경으로 source한 터미널. 새 터미널을 열거나 `source ~/hrc_ws/install/setup.bash`를 다시 실행(8장) |
| sensing이 바로 종료됨: `[FATAL] … IMU camera pose failed: …` | 카메라가 흔들림(정지 검사 실패), Kinect 미연결, udev 권한(2.4). `debug_log/camera_pose_*.yaml`의 `reason` 확인 |
| human_dynamics가 계속 `still waiting for /hrc/camera_pose` | sensing이 아직 IMU 측정 전이거나 종료됨. 또는 두 노드의 `ROS_DOMAIN_ID`/`ROS_LOCALHOST_ONLY`가 다름 |
| `k4a_device_open failed` | 다른 프로세스(k4aviewer, 이전 sensing)가 장치를 잡고 있음. 해당 프로세스를 종료 |
| `ImportError: … libopencv_core.so.4.2` | `sudo ldconfig` 누락, 또는 `/usr/local/lib`가 `LD_LIBRARY_PATH`에 없음(9장) |

## 13. 확인 필요 항목

**본인 확인** (기억·기록이나 직접 실행으로 답할 수 있는 것)
1. SMPL+H `male.npz`의 출처(MANO 사이트의 extended SMPL+H인지)
2. SMPL `SMPL_{MALE,FEMALE}.pkl`의 원본 판(10 shape 판인지, 원래 파일 이름)
3. SMPL-X를 받은 판(로컬 `version.txt`는 v1.0, 원본 README는 v1.1)
4. CLIFF 체크포인트와 `smpl_mean_params.npz`를 받은 곳
5. OpenGL 4.4 / error 207이 어떤 상황에서 났는지(정확한 메시지)
6. ~~chumpy 0.70: 기준 PC `~/.local` 사본이 수정된 것인지~~ → 해결: `__init__.py`의 numpy 별칭 import 한 줄만 주석 처리(6.1, `tools/patch_chumpy.sh`)

**editable 단계** (extern 패키지를 editable로 바꿀 때 확인)
7. editable 설치 결과가 지금 사본 설치와 같은지, pyk4a `k4a_module*.so`가 어디에 생기는지와 git 무시 여부
8. apt nlohmann 3.10.5만으로 pyk4a가 빌드되는지(pyk4a README는 소스 빌드 → `/usr/local`을 안내)
9. ~~chumpy 0.70 PyPI 원본이 numpy 1.26.4에서 import되는지, 설치에 `--no-build-isolation`이 필요한지~~ → 해결: import 안 됨(패치 필요), `--no-build-isolation` 필요(6.1)

**다른 PC 설치 때 확인**
10. Ampere 이후 GPU에서 Body Tracking 1.0이 동작하는지(1장 경고)
11. `libk4a1.3` debconf EULA 미리 동의가 jammy `.deb` 설치에서 동작하는지, `libk4abt1.0`은 수동 동의인지
12. OpenCV 빌드에 필요한 apt 개발 패키지 목록(3장, 기준 PC와 대조 안 함)
13. numpy 1.26.4를 먼저 설치하고 OpenCV 4.2를 빌드하는 순서(기준 PC는 numpy 2.x 상태에서 빌드됨)

**나중**
17. yaw·t 검증 측정(`tools/camera_pose_fit.py`, 10.1). 기준 PC에서도 아직 하지 않았고, yaw 0.0은 미검증 값
18. `tools/k4abt_check.c`로 기준 PC(2080 Ti)의 BT 처리 시간 측정(2.5 통과 기준을 숫자로 정하기 위해)

**이번에 채운 것:** `WITH_TBB=OFF` 이유(5장, Notion 기록), opencv zip sha256 = GitHub archive(5장),
`OMP_WAIT_POLICY=Passive` 이유(9장, 원본 Dockerfile), udev 규칙 출처(2.4), `libk4a1.3` EULA debconf 방법(2.2, 원본 Docker 근거).
