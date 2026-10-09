# ALVR realkk fork

## 目的

realkk 的串流方案：PC（RTX 5090）上跑 KKS VR，經 ALVR 串流到 Quest 3，用色鍵（綠幕）做穿透。
Quest 3 的環境深度（`XR_META_environment_depth`）要從頭盔上行到 PC，給 KKS plugin 做椅子追蹤和遮擋。

上游 ALVR 沒有深度上行。本 fork 在上游基底上套用社群 patch，並修掉審查時發現的三個缺陷。

## 版本與來源

| 項目 | 值 |
|---|---|
| 上游 repo | https://github.com/alvr-org/ALVR （remote `origin`，不 push） |
| 基底 commit | `65ea5b12` feat: Pico 4 headset emulation (#3140)，2026-01-05，版本 21.0.0-dev12，已含 HSV/RGB 色鍵 |
| 分支 `main` | 指向基底 `65ea5b12` |
| 分支 `feat/depth-uplink` | patch 加上本 fork 的修正 |
| patch 來源 | ALVR issue #3202（ColtonMcInroy，2026-06-09）https://github.com/alvr-org/ALVR/issues/3202 ，檔案 https://github.com/user-attachments/files/28731088/ALVR_passthrough_depth.patch |
| fork 版本 | `21.0.0-dev12-realkk.1`，protocol_id 是 `21-dev12-realkk.1` |

patch 在基底上 `git apply --check` 可以乾淨套用，原樣收成一個 commit。

## Commit 清單（`git log --oneline main..feat/depth-uplink`）

| commit | 內容 |
|---|---|
| Feat: Apply community passthrough/depth uplink patch from ALVR #3202 | 原樣套用 patch，內容見下一段 |
| Fix: Send per-view pose and FOV with depth frames | 修正 (a) |
| Fix: Map depth timestamps to the server clock | 修正 (b) |
| Fix: Apply ALVR recentering to relayed depth view poses | 修正 (d)，PLAN-v2 A5 列的項目 |
| Chore: Tag version as 21.0.0-dev12-realkk.1 | 修正 (c) |
| Feat: Add depth relay listener for headset tests | `tools/realkk/depth_listener.py` |
| Docs: Add REALKK.md | 本檔 |

### patch 本體做了什麼

- client：用 raw FFI 實作 `XR_META_environment_depth`。深度 swapchain 的讀回流程是 GLES blit 到 D16，再 CopyImageSubData 到 R16UI，最後 ReadPixels。這樣做是為了繞過 Adreno 740 的深度取樣 bug。讀回後做 LZ4 壓縮，從新的 stream `DEPTH=5` 上行。
- client：Passthrough Camera（Camera2 NDK＋AMediaCodec H264），走 stream `CAMERA=6`。
- server：用 TCP 主動連 `127.0.0.1:<viewer_port>`（預設 9944），把幀轉給外部 viewer。viewer 回送 `MSG_STREAM_CONTROL=100`，可以分別開關 depth 和 camera feed。**預設兩個 feed 都是關的**，viewer 要先送開啟指令。
- 設定：新增 `Video > XR Data Streaming`（預設停用），內有 Environment Depth、Camera Frames、Depth FPS（預設 10）等項目。
- Manifest 加了 `USE_SCENE`、`USE_ANCHOR_API`、`CAMERA`、`PASSTHROUGH_CAMERA_ACCESS`。

### 修正說明

**(a) 每個 view 各帶 pose 和正確的 FOV／intrinsics**
- 原 patch 的問題：
  - header 只放了 `views[0].pose`，命名成 `head_pose`。
  - 欄位 `intrinsics` 裡裝的其實是 `XrFovf` 的四個角度。
- 深度相機不是眼睛相機，右眼深度用左眼 pose 反投影一定會錯。
- 修正後 `DepthFrameHeader` 改成兩個欄位：
  - `view_poses[2]`：client stage space，就是 `xrAcquireEnvironmentDepthImageMETA` 傳回的值。
  - `fov_angles[2]`：弧度，順序是 left、right、up、down。
- 新增 `alvr_packets::fov_to_pinhole_intrinsics()`。影像 y 軸向下時的換算公式：
  - `fx = W / (tan r − tan l)`，`cx = −tan l · fx`
  - `fy = H / (tan u − tan d)`，`cy = tan u · fy`
  - 投影：`u = fx·(x/−z) + cx`，`v = fy·(−y/−z) + cy`（OpenXR 視線方向是 −z）
- relay 改送 `MSG_DEPTH_FRAME_V2 = 3`。舊的 type 1 不再送，避免舊 parser 讀錯欄位卻沒有報錯。

**(b) 時間戳對齊 server 時鐘**
- client 時間戳是 Android 的 XR 時間（monotonic），PC 端沒辦法拿它跟自己的時鐘比對。
- client 多送一個 `client_send_time`：送出前一刻呼叫 `xrGetCurrentTime` 取得。
- server 對每條連線估計 `offset = server_unix_ns − client_xr_ns`。做法是取 10 秒滑動視窗內 `(收到時間 − client_send_time)` 的最小值，也就是單向最小延遲估計。
- 殘差是最小單向延遲，USB 或 Wi-Fi 下通常只有幾 ms。
- V2 header 後面附加三個欄位：`server_timestamp_unix_ns`、`clock_offset_ns`、`server_receive_unix_ns`。
- `client_timestamp` 的語意是「這張深度圖要對應的預測顯示時間」。runtime 不提供拍攝時間，所以 pose 以 header 帶的 view pose 為準。

**(c) 版本 pre tag**
- workspace 版本改成 `21.0.0-dev12-realkk.1`。
- `protocol_id()` 由 major 加 pre tag 組成，所以本 fork 不會跟上游 dev12 的 client／streamer 互連。兩邊的封包格式不同，硬連的話會在序列化時靜默出錯。

**(d) recentering**
- server 轉送前，對兩個 view pose 套用 `TrackingManager::recenter_pose`，和送給 SteamVR 的頭盔、手把 pose 落在同一個空間。
- 預設 stage 設定（不 recenter）時這一步不改變任何值。

### MSG_DEPTH_FRAME_V2 格式（TCP，little-endian）

每則訊息的外框是 `u32 msg_type`、`u32 payload_len`，後面接 payload。depth 的 payload 結構如下：

| offset | 型別 | 欄位 |
|---|---|---|
| 0 | u32 | header_size（目前是 176；新欄位只會往後加，viewer 依這個值跳到像素資料） |
| 4 | u64 | client_timestamp_ns |
| 12 / 40 | f32×7 ×2 | view_pose[0]、view_pose[1]：qx qy qz qw px py pz（recentered stage space，OpenXR 右手系，單位公尺） |
| 68 | u32 | width |
| 72 | u32 | height（兩個 view 上下疊：上半是 view 0，下半是 view 1；row 由上往下） |
| 76 / 80 | f32 | near_z、far_z（far 可能是 +inf） |
| 84 | u32 | format：0 RawD16、1 H264、2 Lz4D16（`lz4_flex::compress_prepend_size`，開頭 4 bytes 是解壓後大小） |
| 88 / 104 | f32×4 ×2 | fov_angles：l r u d（弧度） |
| 120 / 136 | f32×4 ×2 | intrinsics：fx fy cx cy（像素，單一 view 的影像） |
| 152 | u64 | server_timestamp_unix_ns |
| 160 | i64 | clock_offset_ns |
| 168 | u64 | server_receive_unix_ns |
| 176 | u8[] | 像素資料 |

D16 換成線性距離（標準 GL 投影，d ∈ [0,1]）：
- far 有限：`z = 2·n·f / ((f+n) − (2d−1)(f−n))`
- far = inf：`z = n / (1 − d)`

**以下尚未實機驗證，PM 實測時確認：**
- row 方向：patch 有做上下翻轉，是否真的翻成 y 向下。
- 深度編碼：是否就是上面這個標準投影。

驗法見 PLAN-v2 的 R9-P3：深度點雲裡的平面高度對照手把高度。

## Build

前置：工具鏈都裝在 `D:\AIP\.tools`，不改系統 PATH。每次開新的 PowerShell 先 dot-source：

```powershell
. D:\AIP\.tools\alvr-toolchain.ps1
cd D:\AIP\projects\alvr-realkk
git checkout feat/depth-uplink
git submodule update --init --recursive   # openvr headers

cargo xtask build-streamer --release      # -> build\alvr_streamer_windows\
cargo xtask build-client --release        # -> build\alvr_client_android\alvr_client_android.apk
```

工具鏈內容（腳本裡都設好了）：

| 項目 | 版本 | 路徑 |
|---|---|---|
| rustup／cargo／rustc | rustup 1.29.1、rustc 1.99.0 stable，target `aarch64-linux-android` | `.tools\rust\{rustup,cargo}` |
| JDK | Microsoft OpenJDK 17.0.20.1 | `.tools\android\jdk17` |
| Android SDK | cmdline-tools latest、platform-tools 37.0.1、platforms;android-32、build-tools 34.0.0 | `.tools\android\sdk` |
| NDK | r26b（26.1.10909125），和上游 CI 相同 | `.tools\android\sdk\ndk\26.1.10909125` |
| cargo-apk（zarik5 fork）、cargo-ndk 3.5.4、cbindgen | 照 xtask `prepare-deps` 的清單 | `.tools\rust\cargo\bin` |
| libclang 18.1.1（bindgen 用） | 取自 PyPI `libclang` wheel | `.tools\llvm\bin\libclang.dll` |
| ATL 14.44（`atlbase.h`、`atls.lib`） | 本機 VS Build Tools 缺 ATL 元件，從 VS 官方 channel manifest 抽出 Headers 和 X64 套件，sha256 已核對 | `.tools\msvc-atl\atlmfc` |
| CMake | 用 VS 2022 Build Tools 內附的版本 | 系統既有 |

和官方步驟不同的地方：
- **沒有跑 `cargo xtask prepare-deps`**。理由有兩個：它會用 choco 以系統管理員權限安裝套件；Windows 版還會執行 `setx PKG_CONFIG_PATH`，改到使用者環境變數。
- 改成手動準備：
  - `deps\windows\libvpl`：libvpl 2.15.0，用 cmake 建置並安裝到 `alvr_build`。
  - `deps\android_openxr\arm64-v8a\libopenxr_loader.so`：Khronos 1.1.36。
- `deps\` 被 gitignore，重新 clone 時要再準備一次，指令同上。
- APK 只帶 generic loader，Quest 2／3／Pro 可用，Quest 1、Pico、YVR、Lynx 不能用。
- streamer 是**非 GPL** build（沒加 `--gpl`），所以沒有軟體編碼，NVENC 照常可用。

APK 簽章：
- `build-client --release` 第一次執行時，會用 keytool 產生 `build\alvr_client_android\debug.keystore`。參數是 alias `androiddebugkey`，密碼 `alvrclient`，RSA 2048，效期 10000 天。cargo-apk 用這把 keystore 簽 release APK。
- keystore 在 gitignore 裡。之後覆蓋安裝要沿用同一把，刪掉的話要先解除安裝舊 APK。
- package 名稱是 `alvr.client.dev`，和官方 stable（`alvr.client.stable`）不同，兩個可以同時裝在頭盔上。如果頭盔上裝了官方 nightly 版的 `alvr.client.dev`，要先解除安裝。

## PM 實測步驟

前提：Quest 3 已開開發者模式，USB 連 PC，`adb devices` 看得到（adb 在 dot-source 腳本設定的 PATH 裡）。

1. **安裝 APK**
   ```powershell
   . D:\AIP\.tools\alvr-toolchain.ps1
   adb install -r D:\AIP\projects\alvr-realkk\build\alvr_client_android\alvr_client_android.apk
   # 選用：先授權空間資料權限，就不必在頭盔裡點對話框
   adb shell pm grant alvr.client.dev com.oculus.permission.USE_SCENE
   ```
2. **啟動 streamer**：先關掉 Virtual Desktop Streamer（見下一節），再執行 `D:\AIP\projects\alvr-realkk\build\alvr_streamer_windows\ALVR Dashboard.exe`。第一次開會跑設定精靈並註冊 SteamVR driver。
3. **設定**（Dashboard > Settings）：
   - Video > Passthrough：啟用，選 **HSV Chroma Key**（或 RGB Chroma Key）。預設值就是綠色，RGB 預設 (0,255,0)，HSV hue 大約 80–160°。KKS 端把相機的 clearColor 設成純綠 (0,255,0)。
   - Video > XR Data Streaming：**啟用**，Environment Depth 打開。Camera Frames 不需要的話可以關掉，省頻寬也避開相機權限問題。Depth FPS 保持 10。
   - 改完設定要重新連線。XR Data 的設定在連線建立時讀取。
4. **頭盔端**：開啟 ALVR（dev）app。第一次會跳出空間資料（USE_SCENE）權限對話框，選允許。在 Dashboard 按 Trust 配對，然後啟動 SteamVR 串流。
5. **確認深度支援**：
   ```powershell
   adb logcat -c
   adb logcat | Select-String "XR_DIAG|XR_DATA"     # client log tag is "[ALVR NATIVE-RUST]"
   ```
   看到 `[XR_DIAG] Depth system props: supports_depth=1`，就代表 supportsEnvironmentDepth=true。接著應該出現 `Deferred depth provider created successfully!`。
6. **收深度並原樣錄成 tap 檔**：在 PC 另開一個 PowerShell（檔名的日期換成當天）：
   ```powershell
   cd D:\AIP\projects\alvr-realkk\tools\realkk
   python depth_listener.py --seconds 0 --tap D:\AIP\Data\realkk\taps\2026-10-09-room.rktap --mark-key
   ```
   - streamer 每 5 秒重試連線一次，所以要等最多 5 秒。
   - `--seconds 0` 表示一直錄，錄完按 **Ctrl+C** 停止；streamer 斷線時也會自動停。結束時印出 `tap: N messages written`。
   - `--tap` 把收到的每則 relay 訊息連同 PC 收到的時間原樣寫進檔案（含 pose、內參等 header），之後可以離線回放。錄製只需要標準 Python，不用裝 lz4／numpy。格式見 `tools/realkk/rktap.py` 開頭。
   - 讀 socket 和寫檔在不同 thread，寫檔慢不會拖到 relay 的 500 ms 寫入逾時。
   - `--mark-key`：在這個視窗按 **Enter** 就寫入一筆 marker（依序叫 `mark 1`、`mark 2`…；先打字再按 Enter 就用打的字當標籤），畫面會印出 `marker #N ... at <UTC 時間>`。
   - 腳本連上後會自動開啟 depth feed，然後印出：
     - stacked 寬高、單一 view 的寬高、format
     - near／far
     - 兩個 view 的 FOV（度）、fx fy cx cy、pose
     - clock offset
     - 每 50 幀的 fps 和 Mbps
   - 把首幀資訊和最後一行 `total N frames in T s -> X fps` 記下來。P2b 的通過標準是 ≥9 fps。
   - 同時 Dashboard 的 log 會出現 `XR Data: depth frame #1 WxH (...)`。
   - 要存幀的話加 `--dump D:\AIP\Data\realkk\depth_dump`，需要 `pip install numpy lz4`。

   **錄製腳本**（listener 連上、開始印 fps 之後照順序做，全程約 2.5 分鐘）：

   | 步驟 | 動作 | 時間 |
   |---|---|---|
   | 1 | 慢慢轉頭、走動，掃過整個房間 | 60 s |
   | 2 | 站定，盯著椅子不動 | 10 s |
   | 3 | 把椅子移動約 1 m，放好後回到鍵盤按一次 **Enter**（記下 `mark 1`） | — |
   | 4 | 從椅子前方走過去 | 約 5–10 s |
   | 5 | 坐到椅子上，保持不動 | 20 s |
   | 6 | 坐著在面前揮手 | 約 10 s |
   | 7 | 回到鍵盤按 **Ctrl+C** 停止錄製 | — |

   **檢查錄到的內容**（需要 lz4）：
   ```powershell
   uv run --no-project --with lz4 python tap_inspect.py D:\AIP\Data\realkk\taps\2026-10-09-room.rktap
   ```
   - 印出訊息數、各型別數量、時長、depth fps、首幀 header（寬高、near／far、FOV、內參、pose）、marker 列表。
   - `fake frames`：頭盔 readback 失敗時會送出整張都是 0x80 的假幀（解壓後 D16 全為 0x8080），這一行統計有幾張、是第幾幀。比例高就代表 readback 有問題，要回報。
   - 沒有 lz4 時可以加 `--no-decode` 跳過假幀檢查。

   **回放給 viewer（不需要頭盔）**：先關掉 ALVR Dashboard（它也會連 9944），開好要測的 viewer（例如另一個視窗跑 `python depth_listener.py`），再執行：
   ```powershell
   python tap_replay.py D:\AIP\Data\realkk\taps\2026-10-09-room.rktap            # 原速播一次
   python tap_replay.py D:\AIP\Data\realkk\taps\2026-10-09-room.rktap --speed 2  # 兩倍速
   python tap_replay.py D:\AIP\Data\realkk\taps\2026-10-09-room.rktap --loop     # 無限重播，Ctrl+C 停
   ```
   - 行為和 ALVR streamer 一樣：主動連 `127.0.0.1:9944`（`--port` 可改），viewer 沒開就每秒重試。
   - feed 預設全關，viewer 送 `MSG_STREAM_CONTROL` 開啟後才開始播；播放中關掉的 feed，該類訊息直接略過。
   - 訊息依錄製時的間隔送出，內容一個 byte 都不改，header 裡的時間戳仍是錄製當時的值。
   - marker 不會送給 viewer，只在回放視窗印出。
7. **量 overhead**：比較 listener 開啟前和開啟中，Dashboard 統計頁的串流 fps 和 total latency（P2b③：fps 下降 ≤5%）。

## 和 Virtual Desktop 共存

- Quest 上同時只能有一個沉浸式 app。VD 和 ALVR 是兩個獨立 app，要換時從頭盔選單切換。
- PC 端兩邊都是 SteamVR driver，SteamVR 啟動時只用一個 HMD driver。
  - 用 ALVR 前：先完全關閉 Virtual Desktop Streamer（包括系統匣圖示），再開 ALVR Dashboard，最後才啟動 SteamVR。
  - 切回 VD：關掉 ALVR Dashboard 和 SteamVR。如果 SteamVR 還是抓到 ALVR，到 ALVR Dashboard > Installation 取消註冊 driver，或在 SteamVR 設定 > Startup／Manage Add-ons 停用 ALVR driver。
- 這個 fork 的 protocol_id 和官方 ALVR 不同。官方 stable 的 client／streamer 不會連到本 fork；頭盔上可以同時保留官方 stable app 當備案。
- 本 fork 的 streamer 會讀寫 `build\alvr_streamer_windows\` 底下的 session.json。如果另外裝了官方 ALVR，兩邊的設定互不影響，但不要同時開兩個 Dashboard。

## 已知限制與待驗證

- Camera 上行（`CameraFrameHeader`）維持 patch 原樣，只帶單一 pose。realkk 目前用不到。
- 深度 row 方向、D16 編碼、pose 和 SteamVR 座標的對齊，都要用 R9-P3 實機驗證。
- server 會把每張深度幀複製一份成 `ServerCoreEvent::DepthFrame`，丟給 OpenVR event loop，但那邊會直接忽略。這是 patch 原有的行為，有額外記憶體成本，但不影響功能。
- 沒有自動化 build 測試 Android 執行期。目前只驗證過：單元測試（`cargo test -p alvr_packets -p alvr_server_core`）、兩個平台都 build 成功。
